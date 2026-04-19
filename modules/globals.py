# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = True       # Poisson Blending ON by default for seamless edges
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False

# System Configuration
max_memory: int | None = 4           # 4 GB: leaves ~2 GB for macOS on 8 GB machine
execution_providers: List[str] = []  # Detected at startup: CPUExecutionProvider on Intel Mac
execution_threads: int | None = None # Set at startup: 2 (physical cores) on Intel i5 MBP 2017
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles
# GFPGAN ON by default — best quality enhancer for Intel Mac
fp_ui: Dict[str, bool] = {"face_enhancer": True, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 2.2            # Default: Full HD 2-pass unsharp (0.0=off, 2.2=recommended, 5.0=max)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# --- Frame Interpolation ---
# Disabled by default on Intel CPU: blending two frames doubles per-frame cost
enable_interpolation: bool = False  # Temporal smoothing (too slow on CPU-only machines)
interpolation_weight: float = 0     # Blend weight for current frame (0.0-1.0)

# --- Processing Control ---
processing_cancelled: bool = False  # Set True to cancel video processing mid-run

# --- Live FPS Throttle ---
# Intel i5-7267U: cap at 10fps to prevent thermal throttling
target_live_fps: int = 15

# --- Face Detection Quality ---
# Faces with det_score below this are ignored (blurry/partial/far faces)
face_det_score_threshold: float = 0.55

# --- Recent Directories (persisted via switch_states.json) ---
recent_dir_source: str | None = None
recent_dir_target: str | None = None
recent_dir_output: str | None = None

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
