"""Verify the load-time voxel-conditioning adaptation against real pretrained Wan-1.3B (TPU).

    cd ~/maxdiffusion && python ~/verify_cond_load.py

Loads the conditioned transformer from the pretrained base checkpoint via the normal
create_sharded_logical_transformer path (enable_voxel_cond=true) and checks the patch-embed widening:
the patch-embedding is widened to in_channels+raster, the pretrained weights land in [:16] (nonzero),
the new conditioning channels [16:] are exactly zero, and voxel_cond exists.
"""

import sys

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from maxdiffusion import max_utils, pyconfig
from maxdiffusion.pipelines.wan.wan_pipeline import create_sharded_logical_transformer

pyconfig.initialize(
    [sys.argv[0], "src/maxdiffusion/configs/base_wan_1_3b.yml",
     "enable_voxel_cond=true", "run_name=verify_cond_load"]
)
config = pyconfig.config
devices_array = max_utils.create_device_mesh(config)
mesh = Mesh(devices_array, config.mesh_axes)
rngs = nnx.Rngs(0)

print("loading conditioned transformer from pretrained ...")
with mesh:
  model = create_sharded_logical_transformer(devices_array, mesh, rngs, config, subfolder="transformer")

k = model.patch_embedding.kernel.value
new_max = float(jnp.abs(k[..., 16:, :]).max())
base_max = float(jnp.abs(k[..., :16, :]).max())
print("patch_embedding kernel shape:", k.shape)
print(f"base channels [:16] max|w| = {base_max:.4f}   new channels [16:] max|w| = {new_max:.3e}")
print("voxel_cond present:", hasattr(model, "voxel_cond") and model.voxel_cond is not None)
assert k.shape[-2] == 16 + 752, k.shape
assert new_max == 0.0, f"new channels not zero-initialized: {new_max}"
assert base_max > 0.0, "base channels are zero — pretrained weights not loaded into [:16]"
print("COND LOAD OK")
