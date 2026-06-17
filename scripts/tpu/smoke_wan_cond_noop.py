"""No-op invariance smoke test for Wan voxel conditioning (small config, random init, TPU/CPU).

    python scripts/tpu/smoke_wan_cond_noop.py

Builds a small conditioned WanModel, zeros the *new* patch-embedding input channels (the 16:..
slice = the conditioning channels), then runs a forward twice with different proj_ids/proj_depth.
With the conditioning channels zeroed, the output must be identical regardless of the conditioning
input — i.e. the conditioning is inert, exactly the property the load-time patch-embed widening gives
the pretrained model at init. Also exercises the full conditioned forward (concat + widened patch-embed).
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as fnn
from flax import nnx
from jax.sharding import Mesh

from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

CFG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=16,   # inner_dim = 32
    in_channels=16,
    out_channels=16,
    text_dim=32,
    freq_dim=64,
    ffn_dim=64,
    num_layers=1,
    cross_attn_norm=True,
    attention="dot_product",
    scan_layers=False,
    flash_min_seq_length=10**9,   # force dot-product (no flash)
)
COND = dict(
    enable_voxel_cond=True,
    cond_vocab=50,
    cond_feat_dim=8,
    cond_layers=8,
    cond_depth_patch=2,
    cond_depth_stride=2,
    cond_out_dim=4,   # out_layers=(8-2)/2+1=4 -> 16 conditioning channels; patch_in = 16+16 = 32
)


def main():
  print("devices:", jax.devices())
  model = WanModel(rngs=nnx.Rngs(0), **CFG, **COND)

  # Zero the NEW patch-embedding input channels (conditioning slice). Kernel shape:
  # (*patch_size, in_features, out_features) = (1,2,2, 32, 32); channels [16:] are the conditioning.
  k = model.patch_embedding.kernel.value
  print("patch_embedding kernel shape:", k.shape)
  model.patch_embedding.kernel.value = k.at[..., 16:, :].set(0.0)

  b, t, h, w = 1, 4, 8, 8
  hidden = jax.random.normal(jax.random.key(1), (b, CFG["in_channels"], t, h, w))
  timestep = jnp.array([500.0])
  enc = jnp.zeros((b, 4, CFG["text_dim"]))
  l = COND["cond_layers"]
  depth = jax.random.uniform(jax.random.key(2), (b, t, h, w, l)) * 30.0
  ids_zero = jnp.zeros((b, t, h, w, l), jnp.int32)
  ids_rand = jax.random.randint(jax.random.key(3), (b, t, h, w, l), 0, COND["cond_vocab"])

  # maxdiffusion blocks apply logical sharding constraints, which need a mesh in context.
  mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "model"))
  jax.set_mesh(mesh)
  with fnn.logical_axis_rules([]):  # empty rules -> all logical axes replicated (1-device test)
    out_zero = model(hidden, timestep, enc, proj_ids=ids_zero, proj_depth=jnp.zeros_like(depth))
    out_rand = model(hidden, timestep, enc, proj_ids=ids_rand, proj_depth=depth)
  out_zero, out_rand = np.asarray(out_zero), np.asarray(out_rand)

  maxdiff = float(np.abs(out_zero - out_rand).max())
  print("output shape:", out_zero.shape, " expected:", (b, CFG["out_channels"], t, h, w))
  print(f"max |Δ| between zero-cond and random-cond forward: {maxdiff:.3e}")
  assert out_zero.shape == (b, CFG["out_channels"], t, h, w), out_zero.shape
  assert maxdiff < 1e-5, f"conditioning NOT inert: maxdiff={maxdiff}"
  print("NO-OP INVARIANT OK")


if __name__ == "__main__":
  main()
