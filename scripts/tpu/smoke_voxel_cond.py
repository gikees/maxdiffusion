"""Smoke test for the voxel-projection conditioning encoder on the TPU.

    python scripts/tpu/smoke_voxel_cond.py

Constructs WanVoxelProjectionEmbedder and runs a forward on a synthetic id+depth stack, checking
the channel contract (feat 32 + posenc 8 -> Conv1d k6/s4 over 192 -> 47x16 = 752 channels).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from maxdiffusion.models.wan.transformers.transformer_wan import WanVoxelProjectionEmbedder


def main():
  print("devices:", jax.devices())
  enc = WanVoxelProjectionEmbedder(
      rngs=nnx.Rngs(0), vocab=100, feat_dim=32, num_layers=192,
      depth_patch_size=6, depth_stride=4, out_dim=16, num_freqs=8,
  )
  b, t, h, w, ell = 1, 2, 36, 64, 192
  ids = jnp.zeros((b, t, h, w, ell), jnp.int32)
  depth = jnp.zeros((b, t, h, w, ell), jnp.float32)
  out = enc(ids, depth)
  expected = (b, t, h, w, enc.out_layers * enc.out_dim)
  print(f"raster_cond shape: {out.shape}  expected: {expected}  (out_layers={enc.out_layers})")
  assert out.shape == expected, out.shape
  assert enc.out_layers * enc.out_dim == 752, enc.out_layers * enc.out_dim
  print("SMOKE OK")


if __name__ == "__main__":
  main()
