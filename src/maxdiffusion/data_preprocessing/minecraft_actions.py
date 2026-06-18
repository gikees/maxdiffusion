"""Mineflayer action parsing for wander episodes.

Ported from persistent-mp-wm `src/data/minecraft.py` (not importable on the TPU). Converts the per-frame
`action{...}` dicts from an episode JSON into a fixed (T, 25) canonical vector, then strips the dims the
voxel projection already encodes (camera look + player movement), keeping the 16 interaction/UI dims.
"""

import numpy as np

HOTBAR_KEYS_NUM = 9

# Canonical action layout (index -> key).
ACTION_KEYS = [
    "inventory",  # 0
    "ESC",        # 1
    "hotbar.1",   # 2
    "hotbar.2",   # 3
    "hotbar.3",   # 4
    "hotbar.4",   # 5
    "hotbar.5",   # 6
    "hotbar.6",   # 7
    "hotbar.7",   # 8
    "hotbar.8",   # 9
    "hotbar.9",   # 10
    "forward",    # 11
    "back",       # 12
    "left",       # 13
    "right",      # 14
    "jump",       # 15
    "sneak",      # 16
    "sprint",     # 17
    "swapHands",  # 18
    "attack",     # 19
    "use",        # 20
    "pickItem",   # 21
    "drop",       # 22
    "cameraX",    # 23
    "cameraY",    # 24
]

# Strip mouse (cameraX/Y) + movement (forward/back/left/right/jump/sneak/sprint) — the voxel projection
# already reflects camera + movement — and keep the 16 interaction/UI dims (hotbar, attack, use, etc.).
_STRIP_DIMS = {11, 12, 13, 14, 15, 16, 17, 23, 24}
KEEP_DIMS = [i for i in range(len(ACTION_KEYS)) if i not in _STRIP_DIMS]
ACTION_DIM = len(KEEP_DIMS)  # 16


def convert_act_slice_mineflayer(actions):
  """Convert a list of frame dicts (each with an `action` sub-dict) to a (T, 25) one-hot array."""
  out = np.zeros((len(actions), len(ACTION_KEYS)), dtype=np.float32)
  for i, frame in enumerate(actions):
    a = frame["action"]
    for k in ("forward", "back", "left", "right", "jump", "sprint", "sneak", "attack", "use"):
      if a[k]:
        out[i, ACTION_KEYS.index(k)] = 1
    # Mineflayer compound actions -> underlying VPT mouse/keyboard buttons.
    if a["mount"]:
      out[i, ACTION_KEYS.index("use")] = 1          # RMB
    if a["dismount"]:
      out[i, ACTION_KEYS.index("sneak")] = 1        # left shift
    if a["place_block"]:
      out[i, ACTION_KEYS.index("use")] = 1          # RMB
    if a["place_entity"]:
      out[i, ACTION_KEYS.index("use")] = 1          # RMB
    if a["mine"]:
      out[i, ACTION_KEYS.index("attack")] = 1       # LMB
    for h in range(HOTBAR_KEYS_NUM):
      if a.get("hotbar.{}".format(h + 1)):
        out[i, ACTION_KEYS.index("hotbar.{}".format(h + 1))] = 1
    # Camera radians -> degrees (kept in the canonical vector; stripped by KEEP_DIMS).
    out[i, ACTION_KEYS.index("cameraX")] = np.degrees(a["camera"][0])
    out[i, ACTION_KEYS.index("cameraY")] = np.degrees(a["camera"][1])
  return out


def strip_action(arr):
  """(T, 25) canonical -> (T, 16) kept dims (mouse + movement removed)."""
  return arr[:, KEEP_DIMS]
