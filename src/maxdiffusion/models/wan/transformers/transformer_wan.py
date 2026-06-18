"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Tuple, Optional, Dict, Union, Any
import contextlib
import math
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from flax import nnx
import flax.linen as nn
import numpy as np
from .... import common_types
from ...modeling_flax_utils import FlaxModelMixin, get_activation
from ....configuration_utils import ConfigMixin, register_to_config
from ...embeddings_flax import (
    NNXWanImageEmbedding,
    get_1d_rotary_pos_embed,
    NNXFlaxTimesteps,
    NNXTimestepEmbedding,
    NNXPixArtAlphaTextProjection,
)
from ...normalization_flax import FP32LayerNorm
from ...attention_flax import FlaxWanAttention
from ...gradient_checkpoint import GradientCheckpointType

BlockSizes = common_types.BlockSizes


def get_frequencies(max_seq_len: int, theta: int, attention_head_dim: int):
  h_dim = w_dim = 2 * (attention_head_dim // 6)
  t_dim = attention_head_dim - h_dim - w_dim
  freqs = []
  for dim in [t_dim, h_dim, w_dim]:
    freq = get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype=jnp.float32, use_real=False)
    freqs.append(freq)
  freqs = jnp.concatenate(freqs, axis=1)
  t_size = attention_head_dim // 2 - 2 * (attention_head_dim // 6)
  hw_size = attention_head_dim // 6

  dims = [t_size, hw_size, hw_size]

  # Calculate split indices as a static list of integers
  cumulative_sizes = np.cumsum(dims)
  split_indices = cumulative_sizes[:-1].tolist()
  freqs_split = jnp.split(freqs, split_indices, axis=1)
  return freqs_split


class WanRotaryPosEmbed(nnx.Module):

  def __init__(
      self,
      attention_head_dim: int,
      patch_size: Tuple[int, int, int],
      max_seq_len: int,
      theta: float = 10000.0,
  ):
    self.attention_head_dim = attention_head_dim
    self.patch_size = patch_size
    self.max_seq_len = max_seq_len
    self.theta = theta

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    _, num_frames, height, width, _ = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    freqs_split = get_frequencies(self.max_seq_len, self.theta, self.attention_head_dim)

    freqs_f = jnp.expand_dims(jnp.expand_dims(freqs_split[0][:ppf], axis=1), axis=1)
    freqs_f = jnp.broadcast_to(freqs_f, (ppf, pph, ppw, freqs_split[0].shape[-1]))

    freqs_h = jnp.expand_dims(jnp.expand_dims(freqs_split[1][:pph], axis=0), axis=2)
    freqs_h = jnp.broadcast_to(freqs_h, (ppf, pph, ppw, freqs_split[1].shape[-1]))

    freqs_w = jnp.expand_dims(jnp.expand_dims(freqs_split[2][:ppw], axis=0), axis=1)
    freqs_w = jnp.broadcast_to(freqs_w, (ppf, pph, ppw, freqs_split[2].shape[-1]))

    freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
    freqs_final = jnp.reshape(freqs_concat, (1, 1, ppf * pph * ppw, -1))
    return freqs_final


class WanTimeTextImageEmbedding(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      time_freq_dim: int,
      time_proj_dim: int,
      text_embed_dim: int,
      image_embed_dim: Optional[int] = None,
      pos_embed_seq_len: Optional[int] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      flash_min_seq_length: int = 4096,
  ):
    self.timesteps_proj = NNXFlaxTimesteps(dim=time_freq_dim, flip_sin_to_cos=True, freq_shift=0)
    self.time_embedder = NNXTimestepEmbedding(
        rngs=rngs,
        in_channels=time_freq_dim,
        time_embed_dim=dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.act_fn = get_activation("silu")
    self.time_proj = nnx.Linear(
        rngs=rngs,
        in_features=dim,
        out_features=time_proj_dim,
        dtype=jnp.float32,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "embed",
                "mlp",
            ),
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )
    self.text_embedder = NNXPixArtAlphaTextProjection(
        rngs=rngs,
        in_features=text_embed_dim,
        hidden_size=dim,
        act_fn="gelu_tanh",
    )

    self.image_embedder = nnx.data(None)
    if image_embed_dim is not None:
      self.image_embedder = NNXWanImageEmbedding(
          rngs=rngs,
          in_features=image_embed_dim,
          out_features=dim,
          pos_embed_seq_len=pos_embed_seq_len,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          flash_min_seq_length=flash_min_seq_length,
      )

  def __call__(
      self,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      skip_embeddings: bool = False,
  ):
    timestep = self.timesteps_proj(timestep)
    temb = self.time_embedder(timestep)
    with jax.named_scope("time_proj"):
      timestep_proj = self.time_proj(self.act_fn(temb))

    if not skip_embeddings:
      encoder_hidden_states = self.text_embedder(encoder_hidden_states)
      encoder_attention_mask = None
      if encoder_hidden_states_image is not None:
        (
            encoder_hidden_states_image,
            encoder_attention_mask,
        ) = self.image_embedder(encoder_hidden_states_image)
    else:
      encoder_attention_mask = None
      if (
          encoder_hidden_states_image is not None
          and encoder_hidden_states_image.shape[-1] != encoder_hidden_states.shape[-1]
      ):
        img_dim = encoder_hidden_states_image.shape[-1]
        text_dim = encoder_hidden_states.shape[-1]
        if img_dim < text_dim:
          pad_shape = (
              encoder_hidden_states_image.shape[0],
              encoder_hidden_states_image.shape[1],
              text_dim - img_dim,
          )
          encoder_hidden_states_image = jnp.concatenate(
              [
                  encoder_hidden_states_image,
                  jnp.zeros(pad_shape, dtype=encoder_hidden_states_image.dtype),
              ],
              axis=-1,
          )
        else:
          pad_shape = (
              encoder_hidden_states.shape[0],
              encoder_hidden_states.shape[1],
              img_dim - text_dim,
          )
          encoder_hidden_states = jnp.concatenate(
              [encoder_hidden_states, jnp.zeros(pad_shape, dtype=encoder_hidden_states.dtype)], axis=-1
          )

    return (
        temb,
        timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        encoder_attention_mask,
    )


class ApproximateGELU(nnx.Module):
  r"""
  The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
  [paper](https://arxiv.org/abs/1606.08415).
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim_in: int,
      dim_out: int,
      bias: bool,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.proj = nnx.Linear(
        rngs=rngs,
        in_features=dim_in,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "embed",
                "mlp",
            ),
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    with jax.named_scope("gelu"):
      x = self.proj(x)
    return nnx.gelu(x)


class WanFeedForward(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      dim_out: Optional[int] = None,
      mult: int = 4,
      dropout: float = 0.0,
      activation_fn: str = "geglu",
      final_dropout: bool = False,
      inner_dim: int = None,
      bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      enable_jax_named_scopes: bool = False,
  ):
    if inner_dim is None:
      inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim

    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.act_fn = nnx.data(None)
    if activation_fn == "gelu-approximate":
      self.act_fn = ApproximateGELU(
          rngs=rngs,
          dim_in=dim,
          dim_out=inner_dim,
          bias=bias,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    else:
      raise NotImplementedError(f"{activation_fn} is not implemented.")

    self.drop_out = nnx.Dropout(dropout, deterministic=False)
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "mlp",
                "embed",
            ),
        ),
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(
      self,
      hidden_states: jax.Array,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
  ) -> jax.Array:
    hidden_states = self.act_fn(hidden_states)  # Output is (4, 75600, 13824)
    hidden_states = checkpoint_name(hidden_states, "ffn_activation")
    if self.drop_out.rate > 0:
      hidden_states = self.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)
    with jax.named_scope("proj_out"):
      return self.proj_out(hidden_states)  # output is (4, 75600, 5120)


class WanTransformerBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      ffn_dim: int,
      num_heads: int,
      qk_norm: str = "rms_norm_across_heads",
      cross_attn_norm: bool = False,
      eps: float = 1e-6,
      added_kv_proj_dim: Optional[int] = None,
      image_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      dropout: float = 0.0,
      mask_padding_tokens: bool = True,
      enable_jax_named_scopes: bool = False,
      attention_config: Optional[dict] = None,
  ):
    self.enable_jax_named_scopes = enable_jax_named_scopes
    attention_config = {
        "use_base2_exp": False,
        "use_experimental_scheduler": False,
        "ulysses_shards": -1,
        **(attention_config or {}),
    }

    # 1. Self-attention
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    self.attn1 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        is_self_attention=True,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name="self_attn",
        enable_jax_named_scopes=enable_jax_named_scopes,
        attention_config=attention_config,
    )

    # 1. Cross-attention
    self.attn2 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        added_kv_proj_dim=added_kv_proj_dim,
        image_seq_len=image_seq_len,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        is_self_attention=False,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name="cross_attn",
        enable_jax_named_scopes=enable_jax_named_scopes,
        attention_config=attention_config,
    )
    assert cross_attn_norm is True
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=True)

    # 3. Feed-forward
    self.ffn = WanFeedForward(
        rngs=rngs,
        dim=dim,
        inner_dim=ffn_dim,
        activation_fn="gelu-approximate",
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        dropout=dropout,
        enable_jax_named_scopes=enable_jax_named_scopes,
    )
    self.norm3 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)

    key = rngs.params()
    self.adaln_scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 6, dim)) / dim**0.5,
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      temb: jax.Array,
      rotary_emb: jax.Array,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      cached_kv: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
  ):
    with self.conditional_named_scope("transformer_block"):
      # Support both global [B, 6, dim] and per-token [B, seq_len, 6, dim] temb.
      # Per-token temb is used by TI2V where first-frame tokens have timestep=0.
      if temb.ndim == 4:  # Per-token: [B, seq_len, 6, dim]
        adaln = jnp.expand_dims(self.adaln_scale_shift_table, 0)  # [1, 1, 6, dim]
        combined = adaln + temb.astype(jnp.float32)  # [B, seq_len, 6, dim]
        parts = jnp.split(combined, 6, axis=2)
        shift_msa = parts[0].squeeze(2)
        scale_msa = parts[1].squeeze(2)
        gate_msa = parts[2].squeeze(2)
        c_shift_msa = parts[3].squeeze(2)
        c_scale_msa = parts[4].squeeze(2)
        c_gate_msa = parts[5].squeeze(2)
      else:  # Global: [B, 6, dim]
        (
            shift_msa,
            scale_msa,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
        ) = jnp.split(
            (self.adaln_scale_shift_table + temb.astype(jnp.float32)),
            6,
            axis=1,
        )
      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_heads"))
      hidden_states = jax.lax.with_sharding_constraint(hidden_states, axis_names)
      hidden_states = checkpoint_name(hidden_states, "hidden_states")
      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_kv"))
      encoder_hidden_states = jax.lax.with_sharding_constraint(encoder_hidden_states, axis_names)

      # 1. Self-attention
      with self.conditional_named_scope("self_attn"):
        with self.conditional_named_scope("self_attn_norm"):
          norm_hidden_states = (self.norm1(hidden_states.astype(jnp.float32)) * (1 + scale_msa) + shift_msa).astype(
              hidden_states.dtype
          )
        with self.conditional_named_scope("self_attn_attn"):
          attn_output = self.attn1(
              hidden_states=norm_hidden_states,
              encoder_hidden_states=norm_hidden_states,
              rotary_emb=rotary_emb,
              deterministic=deterministic,
              rngs=rngs,
          )
        with self.conditional_named_scope("self_attn_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + attn_output * gate_msa).astype(hidden_states.dtype)

      # 2. Cross-attention
      with self.conditional_named_scope("cross_attn"):
        with self.conditional_named_scope("cross_attn_norm"):
          norm_hidden_states = self.norm2(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
        with self.conditional_named_scope("cross_attn_attn"):
          attn_output = self.attn2(
              hidden_states=norm_hidden_states,
              encoder_hidden_states=encoder_hidden_states,
              deterministic=deterministic,
              rngs=rngs,
              encoder_attention_mask=encoder_attention_mask,
              cached_kv=cached_kv,
          )
        with self.conditional_named_scope("cross_attn_residual"):
          hidden_states = hidden_states + attn_output

      # 3. Feed-forward
      with self.conditional_named_scope("mlp"):
        with self.conditional_named_scope("mlp_norm"):
          norm_hidden_states = (self.norm3(hidden_states.astype(jnp.float32)) * (1 + c_scale_msa) + c_shift_msa).astype(
              hidden_states.dtype
          )
        with self.conditional_named_scope("mlp_ffn"):
          ff_output = self.ffn(norm_hidden_states, deterministic=deterministic, rngs=rngs)
        with self.conditional_named_scope("mlp_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + ff_output.astype(jnp.float32) * c_gate_msa).astype(
              hidden_states.dtype
          )
      return hidden_states

  def compute_kv(
      self,
      encoder_hidden_states: jax.Array,
      encoder_attention_mask: Optional[jax.Array] = None,
  ) -> Dict[str, Tuple[jax.Array, jax.Array]]:
    return self.attn2.compute_kv(encoder_hidden_states, encoder_attention_mask)


class WanVoxelProjectionEmbedder(nnx.Module):
  """Encodes a per-pixel depth-ordered voxel-id + depth stack into latent-grid conditioning channels.

  Input per frame: `ids (B,T,H,W,L)` integer voxel classes (0 = empty) and `depth (B,T,H,W,L)` linear
  camera-space depth. Each id is embedded (learned table), a sinusoidal depth positional encoding is
  concatenated, then a 1D conv compresses the L depth layers (kernel `depth_patch_size`, stride
  `depth_stride`, VALID) to `out_layers = (L - depth_patch_size)//depth_stride + 1`. Output is the
  flattened `(B,T,H,W, out_layers * out_dim)` to channel-concat onto the noisy latent before patch
  embedding. With defaults (L=192, feat=32, freqs=8, k6/s4, out=16): 40 -> 47x16 = 752 channels.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      vocab: int,
      feat_dim: int = 32,
      num_layers: int = 192,
      depth_patch_size: int = 6,
      depth_stride: int = 4,
      out_dim: int = 16,
      num_freqs: int = 8,
      max_period: float = 10000.0,
      depth_scale: float = 1.0,
      use_depth_pos_enc: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.num_freqs = num_freqs
    self.max_period = max_period
    self.depth_scale = depth_scale
    self.use_depth_pos_enc = use_depth_pos_enc
    self.vocab = vocab
    self.out_layers = (num_layers - depth_patch_size) // depth_stride + 1
    self.out_dim = out_dim
    self.dtype = dtype
    self.voxel_embed = nnx.Embed(
        num_embeddings=vocab,
        features=feat_dim,
        rngs=rngs,
        dtype=dtype,
        param_dtype=weights_dtype,
    )
    conv_in = feat_dim + (num_freqs if use_depth_pos_enc else 0)
    self.proj = nnx.Conv(
        conv_in,
        out_dim,
        rngs=rngs,
        kernel_size=(depth_patch_size,),
        strides=(depth_stride,),
        padding="VALID",
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, "conv_out")),
    )

  def _depth_pos_enc(self, depth: jax.Array) -> jax.Array:
    """Sinusoidal encoding of depth -> (..., num_freqs)."""
    half = self.num_freqs // 2
    freqs = jnp.exp(-math.log(self.max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = (depth * self.depth_scale).astype(jnp.float32)[..., None] * freqs
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

  def __call__(self, ids: jax.Array, depth: jax.Array) -> jax.Array:
    b, t, h, w, ell = ids.shape
    # Guard against out-of-vocab ids: an OOB embedding gather yields NaN and poisons the whole run.
    # Real ids should be in-range given cond_voxel_vocab; this is a safety net for future data.
    ids = jnp.clip(ids, 0, self.vocab - 1)
    x = self.voxel_embed(ids)                                      # (B,T,H,W,L,feat)
    if self.use_depth_pos_enc:
      x = jnp.concatenate([x, self._depth_pos_enc(depth).astype(x.dtype)], axis=-1)
    x = x.reshape(b * t * h * w, ell, x.shape[-1])                 # (N, L, conv_in)
    x = self.proj(x)                                               # (N, out_layers, out_dim), VALID
    return x.reshape(b, t, h, w, self.out_layers * self.out_dim)   # (B,T,H,W, out_layers*out_dim)


class WanModel(nnx.Module, FlaxModelMixin, ConfigMixin):

  @register_to_config
  def __init__(
      self,
      rngs: nnx.Rngs,
      model_type="t2v",
      patch_size: Tuple[int] = (1, 2, 2),
      num_attention_heads: int = 40,
      attention_head_dim: int = 128,
      in_channels: int = 16,
      out_channels: int = 16,
      text_dim: int = 4096,
      freq_dim: int = 256,
      ffn_dim: int = 13824,
      num_layers: int = 40,
      dropout: float = 0.0,
      cross_attn_norm: bool = True,
      qk_norm: Optional[str] = "rms_norm_across_heads",
      eps: float = 1e-6,
      image_dim: Optional[int] = None,
      added_kv_proj_dim: Optional[int] = None,
      rope_max_seq_len: int = 1024,
      pos_embed_seq_len: Optional[int] = None,
      image_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      remat_policy: str = "None",
      names_which_can_be_saved: list = [],
      names_which_can_be_offloaded: list = [],
      mask_padding_tokens: bool = True,
      scan_layers: bool = True,
      enable_jax_named_scopes: bool = False,
      attention_config: Optional[dict] = None,
      enable_voxel_cond: bool = False,
      cond_vocab: int = 1,
      cond_feat_dim: int = 32,
      cond_layers: int = 192,
      cond_depth_patch: int = 6,
      cond_depth_stride: int = 4,
      cond_out_dim: int = 16,
      cond_num_freqs: int = 8,
      cond_depth_scale: float = 1.0,
      enable_action_cond: bool = False,
      action_dim: int = 16,
  ):
    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or in_channels
    self.num_layers = num_layers
    self.scan_layers = scan_layers
    self.enable_jax_named_scopes = enable_jax_named_scopes
    attention_config = {
        "use_base2_exp": False,
        "use_experimental_scheduler": False,
        "ulysses_shards": -1,
        **(attention_config or {}),
    }

    # 1. Patch & position embedding
    self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
    # Voxel-projection conditioning is channel-concatenated onto the noisy latent before patch
    # embedding, widening the patch-embed input channels (the new channels are zero-initialized at
    # load time so the pretrained model is undisturbed at init).
    self.enable_voxel_cond = enable_voxel_cond
    self.voxel_cond = nnx.data(None)
    patch_in_channels = in_channels
    if enable_voxel_cond:
      self.voxel_cond = WanVoxelProjectionEmbedder(
          rngs=rngs,
          vocab=cond_vocab,
          feat_dim=cond_feat_dim,
          num_layers=cond_layers,
          depth_patch_size=cond_depth_patch,
          depth_stride=cond_depth_stride,
          out_dim=cond_out_dim,
          num_freqs=cond_num_freqs,
          depth_scale=cond_depth_scale,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
      patch_in_channels = in_channels + self.voxel_cond.out_layers * cond_out_dim
    self.patch_embedding = nnx.Conv(
        patch_in_channels,
        inner_dim,
        rngs=rngs,
        kernel_size=patch_size,
        strides=patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (None, None, None, None, "conv_out"),
        ),
    )

    # 2. Condition embeddings
    # image_embedding_dim=1280 for I2V model
    self.condition_embedder = WanTimeTextImageEmbedding(
        rngs=rngs,
        dim=inner_dim,
        time_freq_dim=freq_dim,
        time_proj_dim=inner_dim * 6,
        text_embed_dim=text_dim,
        image_embed_dim=image_dim,
        pos_embed_seq_len=pos_embed_seq_len,
        flash_min_seq_length=flash_min_seq_length,
    )

    # Action conditioning: a zero-init linear projecting the player action into the time-embedding
    # space, added per-frame to the timestep embedding (AdaLN-zero) so it is inert at init and the
    # pretrained model is undisturbed.
    self.enable_action_cond = enable_action_cond
    self.action_proj = nnx.data(None)
    if enable_action_cond:
      self.action_proj = nnx.Linear(
          action_dim,
          inner_dim,
          rngs=rngs,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          kernel_init=nnx.initializers.zeros,
          bias_init=nnx.initializers.zeros,
      )

    # 3. Transformer blocks
    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(
        in_axes=0,
        out_axes=0,
        transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
    )
    def init_block(rngs):
      return WanTransformerBlock(
          rngs=rngs,
          dim=inner_dim,
          ffn_dim=ffn_dim,
          num_heads=num_attention_heads,
          qk_norm=qk_norm,
          cross_attn_norm=cross_attn_norm,
          eps=eps,
          flash_min_seq_length=flash_min_seq_length,
          flash_block_sizes=flash_block_sizes,
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          attention=attention,
          dropout=dropout,
          mask_padding_tokens=mask_padding_tokens,
          enable_jax_named_scopes=enable_jax_named_scopes,
          added_kv_proj_dim=added_kv_proj_dim,
          image_seq_len=image_seq_len,
          attention_config=attention_config,
      )

    self.gradient_checkpoint = GradientCheckpointType.from_str(remat_policy)
    self.names_which_can_be_offloaded = names_which_can_be_offloaded
    self.names_which_can_be_saved = names_which_can_be_saved
    if scan_layers:
      self.blocks = init_block(rngs)
    else:
      blocks = []
      for _ in range(num_layers):
        block = WanTransformerBlock(
            rngs=rngs,
            dim=inner_dim,
            ffn_dim=ffn_dim,
            num_heads=num_attention_heads,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            image_seq_len=image_seq_len,
            flash_min_seq_length=flash_min_seq_length,
            flash_block_sizes=flash_block_sizes,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
            attention=attention,
            enable_jax_named_scopes=enable_jax_named_scopes,
            attention_config=attention_config,
        )
        blocks.append(block)
      self.blocks = nnx.data(blocks)

    self.norm_out = FP32LayerNorm(rngs=rngs, dim=inner_dim, eps=eps, elementwise_affine=False)
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=out_channels * math.prod(patch_size),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", None)),
    )
    key = rngs.params()
    self.scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 2, inner_dim)) / inner_dim**0.5,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, "embed")),
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def compute_kv_cache(
      self,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      timestep: Optional[jax.Array] = None,
  ) -> Tuple[Dict[str, Tuple[jax.Array, jax.Array]], Optional[jax.Array]]:
    if timestep is None:
      batch_size = encoder_hidden_states.shape[0]
      timestep = jnp.zeros((batch_size,), dtype=jnp.int32)

    with self.conditional_named_scope("condition_embedder"):
      (
          temb,
          timestep_proj,
          encoder_hidden_states,
          encoder_hidden_states_image,
          encoder_attention_mask,
      ) = self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)

    if encoder_hidden_states_image is not None:
      encoder_hidden_states = jnp.concatenate([encoder_hidden_states_image, encoder_hidden_states], axis=1)
      if encoder_attention_mask is not None:
        text_mask = jnp.ones(
            (
                encoder_hidden_states.shape[0],
                encoder_hidden_states.shape[1] - encoder_hidden_states_image.shape[1],
            ),
            dtype=jnp.int32,
        )
        encoder_attention_mask = jnp.concatenate([encoder_attention_mask, text_mask], axis=1)

    if self.scan_layers:

      @nnx.vmap(
          in_axes=(0, None, None),
          out_axes=0,
          transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
      )
      def _compute_kv(block, enc_states, enc_mask):
        return block.compute_kv(enc_states, enc_mask)

      kv_cache = _compute_kv(self.blocks, encoder_hidden_states, encoder_attention_mask)
    else:
      kv_cache_list = []
      for block in self.blocks:
        kv_cache_list.append(block.compute_kv(encoder_hidden_states, encoder_attention_mask))
      keys = kv_cache_list[0].keys()
      kv_cache = {}
      for k in keys:
        k_list = [d[k][0] for d in kv_cache_list]
        v_list = [d[k][1] for d in kv_cache_list]
        kv_cache[k] = (jnp.stack(k_list, axis=0), jnp.stack(v_list, axis=0))

    return kv_cache, encoder_attention_mask

  @jax.named_scope("WanModel")
  def __call__(
      self,
      hidden_states: jax.Array,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      return_dict: bool = True,
      attention_kwargs: Optional[Dict[str, Any]] = None,
      deterministic: bool = True,
      rngs: Optional[nnx.Rngs] = None,
      skip_blocks: Optional[jax.Array] = None,
      cached_residual: Optional[jax.Array] = None,
      return_residual: bool = False,
      kv_cache: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
      rotary_emb: Optional[jax.Array] = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      proj_ids: Optional[jax.Array] = None,
      proj_depth: Optional[jax.Array] = None,
      action: Optional[jax.Array] = None,
  ) -> Union[jax.Array, Tuple[jax.Array, jax.Array], Dict[str, jax.Array]]:
    hidden_states = nn.with_logical_constraint(hidden_states, ("batch", None, None, None, None))
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    with self.conditional_named_scope("rotary_embedding"):
      if rotary_emb is None:
        rotary_emb = self.rope(hidden_states)
    if self.enable_voxel_cond and proj_ids is not None:
      with self.conditional_named_scope("voxel_conditioning"):
        raster_cond = self.voxel_cond(proj_ids, proj_depth)
        hidden_states = jnp.concatenate([hidden_states, raster_cond.astype(hidden_states.dtype)], axis=-1)
    with self.conditional_named_scope("patch_embedding"):
      hidden_states = self.patch_embedding(hidden_states)
      hidden_states = jax.lax.collapse(hidden_states, 1, -1)
    per_token_t = timestep.ndim == 2  # [B, seq_len] for TI2V
    with self.conditional_named_scope("condition_embedder"):
      if per_token_t:
        # Per-token timestep: process time and text embeddings separately.
        # This matches the official WAN 2.2 TI2V pipeline where first-frame
        # tokens receive timestep=0 (clean) and other tokens receive timestep=t.
        bt, sl = timestep.shape
        t_flat = timestep.reshape(-1)  # [B*seq_len]
        t_sinusoidal = self.condition_embedder.timesteps_proj(t_flat)  # [B*sl, freq_dim]
        t_sinusoidal = t_sinusoidal.reshape(bt, sl, -1)  # [B, sl, freq_dim]
        temb = self.condition_embedder.time_embedder(t_sinusoidal)  # [B, sl, dim]
        with jax.named_scope("time_proj"):
          timestep_proj = self.condition_embedder.time_proj(self.condition_embedder.act_fn(temb))  # [B, sl, dim*6]
        timestep_proj = timestep_proj.reshape(bt, sl, 6, -1)  # [B, sl, 6, dim]
        # Text processing
        if kv_cache is None:
          encoder_hidden_states_out = self.condition_embedder.text_embedder(encoder_hidden_states)
        else:
          encoder_hidden_states_out = encoder_hidden_states
        encoder_hidden_states_image_out = None
        encoder_attention_mask_out = None
      else:
        (
            temb,
            timestep_proj,
            encoder_hidden_states_out,
            encoder_hidden_states_image_out,
            encoder_attention_mask_out,
        ) = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            skip_embeddings=(kv_cache is not None),
        )
        timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)

    if self.enable_action_cond and action is not None:
      # Per-frame action -> AdaLN-zero. Add the (zero-init) action projection to the timestep
      # embedding per latent frame, re-project to the 6 AdaLN params, then broadcast each frame
      # across its patch tokens (frame-major, matching the collapse of (T,H,W) above). This uses the
      # per-token AdaLN path; with the zero-init projection it is identical to the global path at
      # init, so the action is inert. Single global timestep for now (diffusion forcing deferred).
      assert not per_token_t, "action conditioning + per-token timestep is not supported yet"
      a = self.action_proj(action.astype(temb.dtype))  # (B, T_lat, dim)
      assert a.shape[1] == post_patch_num_frames, "action frames must match latent frames"
      temb = temb[:, None, :] + a  # (B, T_lat, dim)
      timestep_proj = self.condition_embedder.time_proj(self.condition_embedder.act_fn(temb))
      timestep_proj = timestep_proj.reshape(temb.shape[0], temb.shape[1], 6, -1)  # (B, T_lat, 6, dim)
      reps = post_patch_height * post_patch_width
      temb = jnp.repeat(temb, reps, axis=1)  # (B, seq_len, dim)
      timestep_proj = jnp.repeat(timestep_proj, reps, axis=1)  # (B, seq_len, 6, dim)
      per_token_t = True

    if encoder_attention_mask is None:
      encoder_attention_mask = encoder_attention_mask_out

    if encoder_hidden_states_image_out is not None:
      encoder_hidden_states = jnp.concatenate([encoder_hidden_states_image_out, encoder_hidden_states_out], axis=1)
      if kv_cache is None and encoder_attention_mask is not None:
        text_mask = jnp.ones(
            (
                encoder_hidden_states.shape[0],
                encoder_hidden_states.shape[1] - encoder_hidden_states_image_out.shape[1],
            ),
            dtype=jnp.int32,
        )
        encoder_attention_mask = jnp.concatenate([encoder_attention_mask, text_mask], axis=1)
      encoder_hidden_states = encoder_hidden_states.astype(hidden_states.dtype)
    else:
      encoder_hidden_states = encoder_hidden_states_out.astype(hidden_states.dtype)

    def _run_all_blocks(h):
      if self.scan_layers:

        def scan_fn(carry, block_input):
          hidden_states_carry, rngs_carry = carry
          if kv_cache is not None:
            block, layer_kv_cache = block_input
          else:
            block = block_input
            layer_kv_cache = None

          hidden_states = block(
              hidden_states_carry,
              encoder_hidden_states,
              timestep_proj,
              rotary_emb,
              deterministic,
              rngs_carry,
              encoder_attention_mask,
              cached_kv=layer_kv_cache,
          )
          new_carry = (hidden_states, rngs_carry)
          return new_carry, None

        rematted_block_forward = self.gradient_checkpoint.apply(
            scan_fn,
            self.names_which_can_be_saved,
            self.names_which_can_be_offloaded,
            prevent_cse=not self.scan_layers,
        )
        initial_carry = (h, rngs)

        if kv_cache is not None:
          scan_input = (self.blocks, kv_cache)
        else:
          scan_input = self.blocks

        final_carry, _ = nnx.scan(
            rematted_block_forward,
            length=self.num_layers,
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
        )(initial_carry, scan_input)

        h_out, _ = final_carry
      else:
        h_out = h
        for i, block in enumerate(self.blocks):
          layer_kv_cache = None
          if kv_cache is not None:
            layer_kv_cache = jax.tree.map(lambda x: x[i], kv_cache)

          def layer_forward(hidden_states, l_kv):
            return block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                deterministic,
                rngs,
                encoder_attention_mask=encoder_attention_mask,
                cached_kv=l_kv,
            )

          rematted_layer_forward = self.gradient_checkpoint.apply(
              layer_forward,
              self.names_which_can_be_saved,
              self.names_which_can_be_offloaded,
              prevent_cse=not self.scan_layers,
          )
          h_out = rematted_layer_forward(h_out, layer_kv_cache)
      return h_out

    hidden_states_before_blocks = hidden_states

    if skip_blocks:
      if cached_residual is None:
        raise ValueError("cached_residual must be provided when skip_blocks is True")
      hidden_states = hidden_states + cached_residual
    else:
      hidden_states = _run_all_blocks(hidden_states)

    residual_x = hidden_states - hidden_states_before_blocks

    if per_token_t:
      # temb: [B, seq_len, dim] — per-token modulation for final head
      combined_head = jnp.expand_dims(self.scale_shift_table, 0) + jnp.expand_dims(temb, 2)  # [B, sl, 2, dim]
      shift, scale = jnp.split(combined_head, 2, axis=2)
      shift = shift.squeeze(2)  # [B, sl, dim]
      scale = scale.squeeze(2)  # [B, sl, dim]
    else:
      shift, scale = jnp.split(self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1)
    hidden_states = (self.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift).astype(hidden_states.dtype)
    with jax.named_scope("proj_out"):
      hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
    hidden_states = hidden_states.reshape(batch_size, -1, num_frames, height, width)

    if return_residual:
      return hidden_states, residual_x
    return hidden_states
