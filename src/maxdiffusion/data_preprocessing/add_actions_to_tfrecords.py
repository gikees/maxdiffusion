"""Add a per-frame `actions` field to existing Wan conditioning tfrecords (no VAE re-encode).

The original prep (`prepare_wander_tfrecords.py`) wrote `{latents, proj_ids, proj_depth}` per clip using
non-overlapping windows. This reuses that exact windowing to align actions to the already-encoded latents:
for each episode tfrecord, read its records in order (clip k starts at frame `k*NUM_FRAMES`), take the
first-of-group action per latent frame, strip mouse+movement, and re-write each record with `actions`
(T_lat, 16) added -> a new output dir. Latents/proj bytes are copied verbatim.

    ACT_EPISODES_DIR=<dir with <stem>.json> ACT_TFREC_IN=<existing tfrecords> ACT_TFREC_OUT=<out> \
    [ACT_NUM_FRAMES=33] [ACT_LIMIT=0] python src/maxdiffusion/data_preprocessing/add_actions_to_tfrecords.py
"""

import glob
import json
import os

import numpy as np
import tensorflow as tf

from maxdiffusion.data_preprocessing.minecraft_actions import convert_act_slice_mineflayer, strip_action

# MUST match prepare_wander_tfrecords.py (the windowing that produced the input tfrecords).
NUM_FRAMES = int(os.environ.get("ACT_NUM_FRAMES", "33"))
EPISODES_DIR = os.environ["ACT_EPISODES_DIR"]
TFREC_IN = os.environ["ACT_TFREC_IN"]
TFREC_OUT = os.environ["ACT_TFREC_OUT"]
LIMIT = int(os.environ.get("ACT_LIMIT", "0"))


def first_of_group(t_lat):
  """RGB-frame index feeding each latent frame (Wan causal 4x). Matches prepare_wander_tfrecords.py."""
  return [0] + [4 * (i - 1) + 1 for i in range(1, t_lat)]


def _bytes(t):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(t).numpy()]))


def load_actions(stem):
  with tf.io.gfile.GFile(os.path.join(EPISODES_DIR, stem + ".json"), "r") as f:
    frames = json.load(f)
  return strip_action(convert_act_slice_mineflayer(frames))  # (T, 16)


def main():
  tf.io.gfile.makedirs(TFREC_OUT)
  tfrecs = sorted(tf.io.gfile.glob(os.path.join(TFREC_IN, "*.tfrec")))
  if LIMIT:
    tfrecs = tfrecs[:LIMIT]
  print(f"{len(tfrecs)} tfrecord(s) -> {TFREC_OUT}  (NUM_FRAMES={NUM_FRAMES})", flush=True)

  for path in tfrecs:
    stem = os.path.basename(path)[:-6]  # strip ".tfrec"
    actions = load_actions(stem)  # (T, 16)
    out = os.path.join(TFREC_OUT, stem + ".tfrec")
    n = 0
    with tf.io.TFRecordWriter(out, options=tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
      for raw in tf.data.TFRecordDataset([path], compression_type="GZIP"):
        ex = tf.train.Example.FromString(raw.numpy())
        feat = ex.features.feature
        lat = tf.io.parse_tensor(feat["latents"].bytes_list.value[0], tf.float32)
        t_lat = int(lat.shape[1])
        start = n * NUM_FRAMES
        idx = [min(start + j, actions.shape[0] - 1) for j in first_of_group(t_lat)]
        if start + first_of_group(t_lat)[-1] >= actions.shape[0]:
          print(f"  WARN {stem}: clip {n} action index clamped (T_json={actions.shape[0]})", flush=True)
        act_clip = actions[idx].astype(np.float32)  # (T_lat, 16)
        new = {
            "latents": feat["latents"],
            "proj_ids": feat["proj_ids"],
            "proj_depth": feat["proj_depth"],
            "actions": _bytes(tf.constant(act_clip, tf.float32)),
        }
        writer.write(tf.train.Example(features=tf.train.Features(feature=new)).SerializeToString())
        n += 1
    print(f"  {stem}: {n} clips + actions", flush=True)

  # Read-back check on the last shard.
  ds = tf.data.TFRecordDataset([out], compression_type="GZIP")
  for raw in ds.take(1):
    ex = tf.train.Example.FromString(raw.numpy())
    act = tf.io.parse_tensor(ex.features.feature["actions"].bytes_list.value[0], tf.float32)
    lat = tf.io.parse_tensor(ex.features.feature["latents"].bytes_list.value[0], tf.float32)
    print(f"read-back: actions {act.shape} {act.dtype}, latents {lat.shape} (T_lat match: {act.shape[0]==lat.shape[1]})")
  print("ADD ACTIONS OK", flush=True)


if __name__ == "__main__":
  main()
