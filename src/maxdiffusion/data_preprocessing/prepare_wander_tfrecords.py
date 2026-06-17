"""Encode wander RGB + voxel projections into Wan conditioning tfrecords.

    PREP_IN_DIR=<episodes> PREP_PROJ_DIR=<projections> PREP_OUT_DIR=<out> [PREP_NUM_FRAMES=33] \
    [PREP_LIMIT=1] python src/maxdiffusion/data_preprocessing/prepare_wander_tfrecords.py \
        src/maxdiffusion/configs/base_wan_1_3b.yml height=288 width=512 run_name=prep_wander

For each episode: decode the RGB mp4 (resized to 288x512), Wan-VAE-encode each `num_frames` clip to
`latents (16, T_lat, 36, 64)`, take the projection of the FIRST RGB frame of each VAE temporal group
(`proj_ids (T_lat,36,64,192)` + `proj_depth`), and write a tfrecord per episode with
`{latents, proj_ids, proj_depth}`. Actions are deferred (Phase 2 / action AdaLN). Paths come from env
vars so the config argv stays clean for pyconfig.
"""

import glob
import os
import sys

import cv2
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from flax.linen import partitioning as nn_partitioning

from maxdiffusion import pyconfig
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWanCache
from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline

H, W = 288, 512
NUM_FRAMES = int(os.environ.get("PREP_NUM_FRAMES", "33"))   # -> T_lat = (33-1)//4 + 1 = 9
IN_DIR = os.environ["PREP_IN_DIR"]
PROJ_DIR = os.environ["PREP_PROJ_DIR"]
OUT_DIR = os.environ["PREP_OUT_DIR"]
LIMIT = int(os.environ.get("PREP_LIMIT", "0"))


def _bytes(t):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(t).numpy()]))


def _example(latents, proj_ids, proj_depth):
  feat = {
      "latents": _bytes(tf.constant(np.asarray(latents), tf.float32)),
      "proj_ids": _bytes(tf.constant(np.asarray(proj_ids), tf.int32)),
      "proj_depth": _bytes(tf.constant(np.asarray(proj_depth), tf.float16)),
  }
  return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()


def read_rgb(mp4_path):
  cap = cv2.VideoCapture(mp4_path)
  frames = []
  while True:
    ok, frame = cap.read()
    if not ok:
      break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)  # (H, W, 3)
    frames.append(frame)
  cap.release()
  return np.stack(frames)  # (T, H, W, 3) uint8


def first_of_group(t_lat):
  """RGB-frame index feeding each latent frame (Wan causal 4x: latent 0<-frame 0, i>=1<-4(i-1)+1)."""
  return [0] + [4 * (i - 1) + 1 for i in range(1, t_lat)]


def main():
  pyconfig.initialize(sys.argv)
  config = pyconfig.config
  # Reuse the pipeline's VAE setup (its own mesh with a vae_spatial axis + logical rules).
  comp = WanPipeline._create_common_components(config, vae_only=True)
  vae, vae_mesh = comp["vae"], comp["vae_mesh"]
  vae_rules = config.logical_axis_rules
  mean = jnp.array(vae.latents_mean).reshape(1, 1, 1, 1, vae.z_dim)
  std = jnp.array(vae.latents_std).reshape(1, 1, 1, 1, vae.z_dim)

  tf.io.gfile.makedirs(OUT_DIR)
  mp4s = sorted(glob.glob(os.path.join(IN_DIR, "*.mp4")))
  if LIMIT:
    mp4s = mp4s[:LIMIT]
  print(f"{len(mp4s)} episode(s) -> {OUT_DIR}  ({NUM_FRAMES} frames/clip)", flush=True)

  for mp4 in mp4s:
    stem = os.path.basename(mp4)[:-4]
    proj = np.load(os.path.join(PROJ_DIR, stem + "_projections.npz"))
    ids_all, depth_all = proj["ids"], proj["depth"]   # (T, 36, 64, 192)
    rgb = read_rgb(mp4)                                # (T, H, W, 3) uint8
    t = min(len(rgb), len(ids_all))
    out = os.path.join(OUT_DIR, stem + ".tfrec")
    n = 0
    with tf.io.TFRecordWriter(out) as writer:
      for start in range(0, t - NUM_FRAMES + 1, NUM_FRAMES):
        clip = rgb[start:start + NUM_FRAMES].astype(np.float32) / 127.5 - 1.0   # (F, H, W, 3) in [-1,1]
        with vae_mesh, nn_partitioning.axis_rules(vae_rules):
          lat = vae.encode(jnp.asarray(clip)[None], AutoencoderKLWanCache(vae))[0].mode()
          lat = (lat - mean) / std                     # (1, T_lat, 36, 64, 16)
        lat = np.asarray(lat[0]).transpose(3, 0, 1, 2)  # (16, T_lat, 36, 64)
        t_lat = lat.shape[1]
        idx = [start + j for j in first_of_group(t_lat)]
        writer.write(_example(lat, ids_all[idx].astype(np.int32), depth_all[idx]))
        n += 1
    print(f"  {stem}: T={t} -> {n} clips", flush=True)

  # Read-back check on the last shard.
  ds = tf.data.TFRecordDataset([out])
  for raw in ds.take(1):
    ex = tf.train.Example.FromString(raw.numpy())
    lat = tf.io.parse_tensor(ex.features.feature["latents"].bytes_list.value[0], tf.float32)
    pid = tf.io.parse_tensor(ex.features.feature["proj_ids"].bytes_list.value[0], tf.int32)
    pdp = tf.io.parse_tensor(ex.features.feature["proj_depth"].bytes_list.value[0], tf.float16)
    print(f"read-back: latents {lat.shape} {lat.dtype}, proj_ids {pid.shape} {pid.dtype}, "
          f"proj_depth {pdp.shape} {pdp.dtype}")
  print("PREP OK", flush=True)


if __name__ == "__main__":
  main()
