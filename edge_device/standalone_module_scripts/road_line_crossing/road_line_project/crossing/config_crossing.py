"""Tunable configuration for the road-line crossing system.

This module only holds constants and paths. It imports nothing from torch/cv2 so
it stays cheap to import. The trained model is treated as frozen; nothing here
modifies it.

Values marked "(Step N)" are not used until that build step lands, but are kept
here so all tuning lives in one place.
"""
from __future__ import annotations

from pathlib import Path


# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "training_outputs"
    / "selected_best_model_precision_512_roi"
    / "models"
    / "best_model.pth"
)
DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "debug_outputs"

# A BDD test image directory, used only for offline visual checks (no video yet).
TEST_IMAGE_DIR = PROJECT_ROOT / "processed_dataset" / "final_dataset" / "images" / "test"


# --- Model input geometry --------------------------------------------------
# WAVE 2: input size reduced from 512x288 -> 320x192 for speed on the Pi.
# The ONNX model is re-exported at this size (run export_onnx.py).
INPUT_WIDTH = 320
INPUT_HEIGHT = 192
CROP_TOP_FRACTION = 0.25

# ImageNet normalization (must match training/config.py).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- Class ids -------------------------------------------------------------
CLASS_BACKGROUND = 0
CLASS_SOLID = 1  # restricted_solid_line  (the violation-relevant class)
CLASS_DASHED = 2  # dashed_or_non_restricted_line

CLASS_NAMES = {
    CLASS_BACKGROUND: "background",
    CLASS_SOLID: "restricted_solid_line",
    CLASS_DASHED: "dashed_or_non_restricted_line",
}

# Overlay colors (R, G, B) for visual debugging.
OVERLAY_COLORS = {
    CLASS_SOLID: (255, 40, 40),    # red   = restricted solid
    CLASS_DASHED: (40, 120, 255),  # blue  = dashed / non-restricted
}
OVERLAY_ALPHA = 0.5


# --- Source ROI crop (for raw video frames) --------------------------------
# Phone clips here are square 1440x1440 with lots of sky; feeding them whole would
# squish the road. We first crop a landscape road band from each raw frame:
# (top, bottom, left, right) as fractions of the frame.
# WAVE 2: also crop the LEFT 25% away. In Sri Lanka (drive on the left) the
# solid/dashed centre line the vehicle can cross is on the RIGHT, so the left edge
# of the frame is the least useful region. Dropping it cuts pixels AND removes the
# left road-edge line that previously caused a false positive.
# NOTE: this horizontal crop moves the vehicle's forward path away from frame
# centre, so EGO_CENTER_X and the hysteresis zones below are remapped to match.
SOURCE_CROP = (0.40, 0.96, 0.25, 1.0)
# The source crop already removes the sky, so override the model's own top-crop.
# None keeps whatever the checkpoint was trained with.
SOURCE_CROP_MODEL_TOP = 0.0


# --- Inference -------------------------------------------------------------
# If set (0..1), a pixel is called SOLID only when its class-1 probability is at
# least this value; otherwise the per-pixel argmax is used. Threshold sweep on
# the selected model suggests 0.5 default, 0.7-0.8 to favor precision.
# None => plain argmax (matches how the model was evaluated).
SOLID_CONF_THRESHOLD = None


# --- Near-field band (Step 2) ---------------------------------------------
# Fraction of the (already top-cropped) model frame height to keep, measured
# from the bottom. 0.30 => keep the bottom 30%.
NEAR_FIELD_BAND_FRACTION = 0.30
# Connected components smaller than this many pixels are treated as noise.
MIN_BLOB_AREA = 60
# Morphological closing kernel size (px) to bridge small gaps in broken lines.
MORPH_CLOSE_KERNEL = 5


# --- Crossing logic (Step 4) ----------------------------------------------
# WAVE 2: the left 25% of the frame is cropped (SOURCE_CROP left=0.25), so the kept
# width is [0.25, 1.0] of the original. A full-frame x maps to (x-0.25)/0.75 in the
# cropped/model frame. The vehicle centre (full 0.50) -> 0.333; the old zones
# (full 0.42 / 0.58) -> 0.227 / 0.44. These keep the crossing geometry correct.
EGO_CENTER_X = 0.333         # vehicle forward path, remapped into the left-cropped frame
HYSTERESIS_RIGHT = 0.44      # "clearly right" zone boundary (remapped)
HYSTERESIS_LEFT = 0.227      # "clearly left" zone boundary (remapped)
CONFIRM_FRAMES = 4           # consecutive frames required to confirm a crossing
COOLDOWN_FRAMES = 15         # WAVE 2: ~3s at 5 FPS (was 30 frames at 10 FPS)

# Reject a frame-to-frame jump larger than this as a line-switch artifact rather
# than real motion. WAVE 2: raised from 0.25 -> 0.33 because the narrower (0.75x)
# cropped frame and the lower 5 FPS both make legitimate per-frame motion larger.
MAX_POSITION_JUMP = 0.33
# A big jump must persist for at least this many frames to count as a genuine
# line-switch (re-baseline). A shorter spike is treated as transient noise and
# ignored, so a single bad frame can't reset a crossing mid-sweep.
JUMP_CONFIRM_FRAMES = 2
# Require the tracked line to actually pass through the center (dead band) during a
# crossing. Kills "teleport" false positives that skip the middle entirely.
REQUIRE_CENTER_PASSAGE = True
