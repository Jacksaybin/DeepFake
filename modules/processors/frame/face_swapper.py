from typing import Any, List, Optional
import cv2
import insightface
from insightface.utils import face_align
import threading
import numpy as np
import platform
import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid
from modules.gpu_processing import gpu_gaussian_blur, gpu_sharpen, gpu_add_weighted, gpu_resize, gpu_cvt_color
import os
import onnxruntime
from collections import deque
import time

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"

# --- START: Added for Interpolation ---
PREVIOUS_FRAME_RESULT = None # Stores the final processed frame from the previous step
# --- END: Added for Interpolation ---

# Platform detection
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'
IS_INTEL_MAC = platform.system() == 'Darwin' and platform.machine() == 'x86_64'

# Face detection cache (used on all platforms for live mode)
FACE_DETECTION_CACHE = {}
LAST_DETECTION_TIME = 0.0
# Intel i5-7267U processes ~6-8fps live. Re-detect every ~150ms (every frame
# at 6fps) instead of 100ms to avoid back-to-back detections wasting CPU.
DETECTION_INTERVAL = 0.033 if IS_APPLE_SILICON else 0.15

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

def pre_check() -> bool:
    # Use models_dir instead of abs_dir to save to the correct location
    download_directory_path = models_dir
    
    # Make sure the models directory exists, catch permission errors if they occur
    try:
        os.makedirs(download_directory_path, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory {download_directory_path} due to permission error: {e}")
        return False
    
    # Use the direct download URL from Hugging Face (FP32 model for broad GPU compatibility)
    conditional_download(
        download_directory_path,
        [
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
        ],
    )
    return True


def pre_start() -> bool:
    # Check for either model variant
    fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
    if not os.path.exists(fp16_path) and not os.path.exists(fp32_path):
        update_status("Model not found! Please download inswapper_128.onnx.", NAME)
        print(f"Directory: {models_dir}")
        return False

    # Try to get the face swapper to ensure it loads correctly
    if get_face_swapper() is None:
        # Error message already printed within get_face_swapper
        return False

    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # Prefer FP32 for broad GPU compatibility (FP16 can produce NaN
            # on GPUs without Tensor Cores, e.g. GTX 16xx).  Fall back to
            # FP16 when FP32 is not available.
            fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
            fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
            if os.path.exists(fp32_path):
                model_path = fp32_path
            elif os.path.exists(fp16_path):
                model_path = fp16_path
            else:
                update_status(f"No inswapper model found in {models_dir}.", NAME)
                return None
            update_status("Loading face swapper model...", NAME)
            print(f"Path: {model_path}")
            try:
                # Use shared provider config (handles Intel Mac CPU-only correctly)
                from modules.processors.frame._onnx_enhancer import build_provider_config
                providers_config = build_provider_config()

                # Build session options for optimal performance
                session_options = onnxruntime.SessionOptions()
                session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

                cpu_count = max(1, os.cpu_count() or 1)  # 4 on Intel Core i5
                session_options.intra_op_num_threads = cpu_count
                # inter_op = 1: THREAD_SEMAPHORE=1 ensures only 1 ONNX call runs
                # at a time, so parallel inter-op just adds overhead
                session_options.inter_op_num_threads = 1
                session_options.enable_mem_pattern = True
                session_options.enable_mem_reuse = True
                # Prevent idle spin-wait from burning CPU cycles between frames
                session_options.add_session_config_entry(
                    'session.intra_op.allow_spinning', '0'
                )
                session_options.add_session_config_entry(
                    'session.inter_op.allow_spinning', '0'
                )

                FACE_SWAPPER = insightface.model_zoo.get_model(
                    model_path,
                    providers=providers_config,
                    sess_options=session_options
                )
                
                # Warmup the model to trigger JIT/compilation
                try:
                    dummy_face = np.zeros((128, 128, 3), dtype=np.uint8)
                    dummy_latent = np.zeros((1, 512), dtype=np.float32)
                    # Use internal session for a quick dummy run if possible, 
                    # or just a tiny inference call
                    if hasattr(FACE_SWAPPER, 'session'):
                        input_feed = {
                            inp.name: np.zeros(
                                [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
                                dtype=np.float32
                            ) for inp in FACE_SWAPPER.session.get_inputs()
                        }
                        FACE_SWAPPER.session.run(None, input_feed)
                except Exception as warmup_e:
                    print(f"{NAME}: Warmup skipped: {warmup_e}")

                update_status("Face swapper model loaded successfully.", NAME)
                print(f"Threads: {cpu_count}")
            except Exception as e:
                update_status("Error loading face swapper model!", NAME)
                print(f"Error: {e}")
                FACE_SWAPPER = None
                return None
    return FACE_SWAPPER


def _build_face_hull_mask(target_img: Frame, target_face: Face) -> np.ndarray:
    """Build a precise convex-hull face mask from 106 facial landmarks.

    Using the full landmark set instead of a plain white 128×128 rectangle
    gives a tighter fit around jawline, temples and forehead — preventing
    the swapped face from bleeding into hair or background.

    Returns a float32 [0,1] mask the same size as *target_img*.
    """
    h, w = target_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    lm = None
    if hasattr(target_face, 'landmark_2d_106') and target_face.landmark_2d_106 is not None:
        lm = target_face.landmark_2d_106.astype(np.float32)
    elif hasattr(target_face, 'kps') and target_face.kps is not None:
        lm = target_face.kps.astype(np.float32)   # fallback: 5-pt

    if lm is None or len(lm) < 5:
        return mask

    pts = np.clip(lm, [0, 0], [w - 1, h - 1]).astype(np.int32)

    if len(pts) >= 106:
        # Face contour = landmarks 0-32 (jaw)
        jaw = pts[0:33]
        # Eyebrows: 33-42 (left) + 43-52 (right)
        brows = pts[33:53]
        # Estimate forehead: push brow-center upward by chin→brow distance
        chin_pt = pts[16]
        brow_center = brows.mean(axis=0)
        up = brow_center - chin_pt
        up_len = float(np.linalg.norm(up))
        if up_len > 0:
            up_norm = up / up_len
            # push 80% of chin→brow distance above brows
            forehead = (brows + up_norm * up_len * 0.8).astype(np.int32)
            forehead = np.clip(forehead, [0, 0], [w - 1, h - 1])
            all_pts = np.concatenate([jaw, forehead], axis=0)
        else:
            all_pts = jaw
    else:
        all_pts = pts

    hull = cv2.convexHull(all_pts)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1.0)

    # Compute face size for adaptive blur
    face_area = float(np.sum(mask > 0))
    face_r = max(1, int(np.sqrt(face_area / np.pi)))   # approximate radius

    # Erode a small amount to pull mask away from face boundary (avoids
    # sharp seams at jaw / temples), then feather the edge softly.
    erode_r = max(2, face_r // 20)          # tiny: ~5% of face radius
    blur_r  = max(5, face_r // 10)          # moderate feather
    k_erode = 2 * erode_r + 1
    k_blur  = 2 * blur_r  + 1

    mask_u8 = (mask * 255).astype(np.uint8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
    mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
    mask_u8 = cv2.GaussianBlur(mask_u8, (k_blur, k_blur), 0)
    return mask_u8.astype(np.float32) / 255.0


def _fast_paste_back(target_img: Frame, bgr_fake: np.ndarray, aimg: np.ndarray,
                     M: np.ndarray, target_face: Face = None) -> Frame:
    """Paste swapped face back onto target with landmark-precise masking.

    Improvements over the original:
    - Optionally uses a 106-landmark convex-hull mask instead of a plain
      white rectangle, giving a tight face boundary so eyes, nose and mouth
      edges align precisely.
    - Erode kernel is *elliptical* (matches face shape) instead of square.
    - Erode amount is proportional to face area (not just the face size
      average), so small faces in high-res frames don't get over-eroded.
    - Blur sigma is separated from kernel size for smoother falloff.
    """
    h, w = target_img.shape[:2]
    IM = cv2.invertAffineTransform(M)

    # Warp swapped face to full-frame coordinates (LANCZOS4: sharper at FullHD)
    bgr_fake_full = cv2.warpAffine(bgr_fake, IM, (w, h),
                                   flags=cv2.INTER_LANCZOS4, borderValue=0.0)

    # --- Mask strategy ---
    # If we have 106 landmarks use the landmark-precise hull mask.
    # Otherwise fall back to the warped white-rectangle approach.
    has_lm106 = (target_face is not None and
                 hasattr(target_face, 'landmark_2d_106') and
                 target_face.landmark_2d_106 is not None and
                 len(target_face.landmark_2d_106) >= 106)

    if has_lm106:
        mask_f = _build_face_hull_mask(target_img, target_face)   # float32 [0,1]
    else:
        # Fallback: warp the plain white crop mask
        img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
        img_white_full = cv2.warpAffine(img_white, IM, (w, h), borderValue=0.0)

        # Tight bounding box for crop-based processing
        nonzero = np.where(img_white_full > 20)
        if len(nonzero[0]) == 0:
            return target_img
        y1, y2 = int(nonzero[0].min()), int(nonzero[0].max())
        x1, x2 = int(nonzero[1].min()), int(nonzero[1].max())

        mask_size = int(np.sqrt((y2 - y1) * (x2 - x1)))
        k_erode = max(3, mask_size // 14)     # slightly tighter than before
        k_blur  = max(5, mask_size // 20)
        pad = k_erode + k_blur + 2
        y1p, y2p = max(0, y1 - pad), min(h, y2 + pad + 1)
        x1p, x2p = max(0, x1 - pad), min(w, x2 + pad + 1)

        mask_crop = img_white_full[y1p:y2p, x1p:x2p]
        mask_crop = np.where(mask_crop > 20, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_erode, k_erode))    # ellipse, not square
        mask_crop = cv2.erode(mask_crop, kernel, iterations=1)
        mask_crop = cv2.GaussianBlur(mask_crop, (2 * k_blur + 1, 2 * k_blur + 1), 0)

        mask_f = np.zeros((h, w), dtype=np.float32)
        mask_f[y1p:y2p, x1p:x2p] = mask_crop.astype(np.float32) / 255.0

    # --- Determine tight crop region for blend (avoid processing whole frame) ---
    nonzero_m = np.where(mask_f > 1e-3)
    if len(nonzero_m[0]) == 0:
        return target_img
    ry1, ry2 = int(nonzero_m[0].min()), int(nonzero_m[0].max())
    rx1, rx2 = int(nonzero_m[1].min()), int(nonzero_m[1].max())
    ry2 = min(h, ry2 + 1)
    rx2 = min(w, rx2 + 1)

    mask_3d    = mask_f[ry1:ry2, rx1:rx2, np.newaxis]          # (H,W,1)
    fake_crop  = bgr_fake_full[ry1:ry2, rx1:rx2].astype(np.float32)
    tgt_crop   = target_img[ry1:ry2, rx1:rx2].astype(np.float32)

    blended = fake_crop * mask_3d + tgt_crop * (1.0 - mask_3d)

    result = target_img.copy()
    result[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Precision face swapping: landmark-aligned paste-back + skin-only colour correction."""
    face_swapper = get_face_swapper()
    if face_swapper is None:
        update_status("Face swapper model not loaded or failed to load. Skipping swap.", NAME)
        return temp_frame

    # Safety check for faces
    if source_face is None or target_face is None:
        return temp_frame
    if not hasattr(source_face, 'normed_embedding') or source_face.normed_embedding is None:
        return temp_frame

    # Only copy original frame when we'll actually need it
    opacity = max(0.0, min(1.0, getattr(modules.globals, "opacity", 1.0)))
    mouth_mask_enabled = getattr(modules.globals, "mouth_mask", False)
    need_original = opacity < 1.0 or mouth_mask_enabled or True  # always keep for LAB
    original_frame = temp_frame.copy()

    if temp_frame.dtype != np.uint8:
        temp_frame = np.clip(temp_frame, 0, 255).astype(np.uint8)

    try:
        if not temp_frame.flags['C_CONTIGUOUS']:
            temp_frame = np.ascontiguousarray(temp_frame)

        # Use paste_back=False and our landmark-precise paste-back
        if any("DmlExecutionProvider" in p for p in modules.globals.execution_providers):
            with modules.globals.dml_lock:
                bgr_fake, M = face_swapper.get(
                    temp_frame, target_face, source_face, paste_back=False
                )
        else:
            bgr_fake, M = face_swapper.get(
                temp_frame, target_face, source_face, paste_back=False
            )

        if bgr_fake is None or not isinstance(bgr_fake, np.ndarray):
            return original_frame

        # Get aligned crop for fallback mask geometry
        aimg, _ = face_align.norm_crop2(temp_frame, target_face.kps, face_swapper.input_size[0])

        # Pass target_face so _fast_paste_back can use 106-landmark hull mask
        swapped_frame = _fast_paste_back(temp_frame, bgr_fake, aimg, M,
                                         target_face=target_face)
        swapped_frame = np.clip(swapped_frame, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"Error during face swap: {e}")
        return original_frame

    # --- Post-swap Processing (Masking, Opacity, etc.) ---
    # Now, work with the guaranteed uint8 'swapped_frame'

    if mouth_mask_enabled: # Check if mouth_mask is enabled
        # Create a mask for the target face
        face_mask = create_face_mask(target_face, original_frame) # Use original_frame for mask creation geometry

        # Create the mouth mask using the ORIGINAL frame (before swap) for cutout
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, original_frame) # Use original_frame for real mouth cutout
        )

        # Apply the mouth area only if mouth_cutout exists
        if mouth_cutout is not None and mouth_box != (0,0,0,0):
            # Apply mouth area (from original) onto the 'swapped_frame'
            swapped_frame = apply_mouth_area(
                swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
            )

            # Draw bounding box only while slider is being dragged
            if getattr(modules.globals, "show_mouth_mask_box", False):
                mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
                swapped_frame = draw_mouth_mask_visualization(
                    swapped_frame, target_face, mouth_mask_data
                )
        
    # --- Poisson Blending ---
    if getattr(modules.globals, "poisson_blend", False):
        face_mask = create_face_mask(target_face, temp_frame)
        if face_mask is not None:
            y_indices, x_indices = np.where(face_mask > 0)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
                src_crop  = swapped_frame[y_min : y_max + 1, x_min : x_max + 1]
                mask_crop = face_mask[y_min : y_max + 1, x_min : x_max + 1]
                try:
                    swapped_frame = cv2.seamlessClone(
                        src_crop, original_frame, mask_crop, center, cv2.NORMAL_CLONE,
                    )
                except Exception as e:
                    print(f"Poisson blending failed: {e}")
    else:
        # --- Skin-only LAB Colour Correction ---
        # Apply colour correction ONLY to skin-coloured pixels inside the face
        # bbox. This prevents the colour stats from being skewed by dark eye
        # regions or bright teeth, and stops the correction from tinting the
        # eyes / lips.
        try:
            if hasattr(target_face, 'bbox') and target_face.bbox is not None:
                bx1, by1, bx2, by2 = [int(v) for v in target_face.bbox]
                h_f, w_f = swapped_frame.shape[:2]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w_f, bx2), min(h_f, by2)
                if bx2 > bx1 and by2 > by1:
                    src_roi = swapped_frame[by1:by2, bx1:bx2].astype(np.float32) / 255.0
                    tgt_roi = original_frame[by1:by2, bx1:bx2].astype(np.float32) / 255.0

                    src_lab = cv2.cvtColor(src_roi, cv2.COLOR_BGR2LAB)
                    tgt_lab = cv2.cvtColor(tgt_roi, cv2.COLOR_BGR2LAB)

                    # --- Build a skin mask in LAB space ---
                    # Skin pixels: L in [20,85] (not too dark/bright),
                    #              a in [128+3, 128+25] (slight reddish),
                    #              b in [128+5, 128+28] (slight yellowish)
                    # (LAB values from cv2 float32 are: L∈[0,100], a/b∈[-127,127]
                    #  but from float [0,1] BGR→LAB the ranges differ;
                    #  for uint8 input *scaled to [0,1]*: L∈[0,100], a/b∈[-127,127])
                    L  = src_lab[:, :, 0]   # 0-100
                    a  = src_lab[:, :, 1]   # -127 to +127 (green/red)
                    b  = src_lab[:, :, 2]   # -127 to +127 (blue/yellow)

                    # Skin: moderate lightness, red-shifted, yellow-shifted
                    skin_mask = (
                        (L  > 20) & (L  < 88) &
                        (a  > 2)  & (a  < 28) &
                        (b  > 4)  & (b  < 30)
                    ).astype(np.float32)

                    # Soft-erode / blur the skin mask to avoid harsh borders
                    sk_u8 = (skin_mask * 255).astype(np.uint8)
                    sk_u8 = cv2.GaussianBlur(sk_u8, (7, 7), 0)
                    skin_mask = sk_u8.astype(np.float32) / 255.0

                    # Compute stats only over skin pixels
                    skin_px_count = float(np.sum(skin_mask > 0.1))
                    if skin_px_count > 50:   # enough pixels for meaningful stats
                        # Weighted mean / std
                        w3 = skin_mask[:, :, np.newaxis]

                        def _wstats(ch):
                            vals = ch * skin_mask
                            n = skin_px_count
                            mn = np.sum(vals) / n
                            std = float(np.sqrt(np.sum(skin_mask * (ch - mn) ** 2) / n))
                            return mn, max(std, 1e-6)

                        s_mean_L, s_std_L = _wstats(src_lab[:,:,0])
                        t_mean_L, t_std_L = _wstats(tgt_lab[:,:,0])
                        s_mean_a, s_std_a = _wstats(src_lab[:,:,1])
                        t_mean_a, t_std_a = _wstats(tgt_lab[:,:,1])
                        s_mean_b, s_std_b = _wstats(src_lab[:,:,2])
                        t_mean_b, t_std_b = _wstats(tgt_lab[:,:,2])

                        corrected = src_lab.copy()
                        # Correct all 3 channels on skin-covered pixels
                        corrected[:,:,0] = s_mean_L + (src_lab[:,:,0] - s_mean_L) * (t_std_L / s_std_L) * 0.5 + t_mean_L * 0.5 - s_mean_L * 0.5
                        corrected[:,:,1] = (src_lab[:,:,1] - s_mean_a) * (t_std_a / s_std_a) + t_mean_a
                        corrected[:,:,2] = (src_lab[:,:,2] - s_mean_b) * (t_std_b / s_std_b) + t_mean_b

                        corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
                        corrected_bgr = np.clip(corrected_bgr * 255.0, 0, 255).astype(np.uint8)

                        # Blend: only apply correction where skin mask is strong
                        # (mắt/môi có skin_mask ~ 0 nên không bị ảnh hưởng)
                        blend_mask = skin_mask[:, :, np.newaxis]  # [0,1]
                        roi_f  = swapped_frame[by1:by2, bx1:bx2].astype(np.float32)
                        corr_f = corrected_bgr.astype(np.float32)
                        # 65% corrected on skin + 35% original to avoid over-correction
                        fused = roi_f + blend_mask * (corr_f - roi_f) * 0.65
                        swapped_frame[by1:by2, bx1:bx2] = np.clip(fused, 0, 255).astype(np.uint8)
        except Exception:
            pass   # Non-critical — never crash the swap
        
    # Apply opacity blend between the original frame and the swapped frame
    if opacity >= 1.0:
        return swapped_frame.astype(np.uint8)

    # Blend the original_frame with the (potentially mouth-masked) swapped_frame
    final_swapped_frame = gpu_add_weighted(original_frame.astype(np.uint8), 1 - opacity, swapped_frame.astype(np.uint8), opacity, 0)
    return final_swapped_frame.astype(np.uint8)


def get_faces_optimized(frame: Frame, use_cache: bool = True) -> Optional[List[Face]]:
    """Face detection with time-based cache for live mode on all platforms.

    On Intel Mac re-detects every ~100ms (DETECTION_INTERVAL=0.10) instead of
    every frame, saving ~30-40ms of CPU per skipped detection.
    """
    global LAST_DETECTION_TIME, FACE_DETECTION_CACHE

    if use_cache:
        current_time = time.time()
        if (current_time - LAST_DETECTION_TIME) < DETECTION_INTERVAL and FACE_DETECTION_CACHE:
            return FACE_DETECTION_CACHE.get('faces')
        LAST_DETECTION_TIME = current_time

    if modules.globals.many_faces:
        faces = get_many_faces(frame)
    else:
        face = get_one_face(frame)
        faces = [face] if face else None

    if use_cache:
        FACE_DETECTION_CACHE['faces'] = faces

    return faces

# --- START: Helper function for interpolation and sharpening ---
def apply_post_processing(current_frame: Frame, swapped_face_bboxes: List[np.ndarray]) -> Frame:
    """Full HD sharpening pipeline: 2-pass unsharp masking on each swapped face region.

    Pass 1 (fine):   sigma=0.8 recovers skin texture and hair detail.
    Pass 2 (coarse): sigma=2.0 boosts local contrast (edges, eyelids, lips).
    Both passes are scaled by the 'sharpness' slider (0.0–5.0).
    Temporal interpolation follows if enabled.
    """
    global PREVIOUS_FRAME_RESULT

    processed_frame = current_frame.copy()
    sharpness_value = getattr(modules.globals, "sharpness", 0.0)

    if sharpness_value > 0.0 and swapped_face_bboxes:
        height, width = processed_frame.shape[:2]

        for bbox in swapped_face_bboxes:
            if not hasattr(bbox, '__iter__') or len(bbox) != 4:
                continue
            try:
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(width, int(bbox[2])), min(height, int(bbox[3]))
            except (ValueError, TypeError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            roi = processed_frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_f = roi.astype(np.float32)

            # --- Pass 1: fine detail (skin texture, pores, fine hair) ---
            # sigma=0.8 ≈ 1-pixel radius → targets sub-pixel sharpness
            fine_amount  = sharpness_value * 0.30   # up to 1.5× at slider=5
            blur_fine    = cv2.GaussianBlur(roi_f, (0, 0), sigmaX=0.8)
            roi_f = roi_f + fine_amount * (roi_f - blur_fine)

            # --- Pass 2: local contrast (edges, eyelids, lip boundary) ---
            # sigma=2.5 ≈ 5-pixel radius → targets macro sharpness
            coarse_amount = sharpness_value * 0.20  # up to 1.0× at slider=5
            blur_coarse   = cv2.GaussianBlur(roi_f, (0, 0), sigmaX=2.5)
            roi_f = roi_f + coarse_amount * (roi_f - blur_coarse)

            processed_frame[y1:y2, x1:x2] = np.clip(roi_f, 0, 255).astype(np.uint8)

    # Temporal interpolation (disabled by default on CPU-only machines)
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)
    interpolation_weight = getattr(modules.globals, "interpolation_weight", 0.2)
    final_frame = processed_frame

    if enable_interpolation and 0 < interpolation_weight < 1:
        if (PREVIOUS_FRAME_RESULT is not None and
                PREVIOUS_FRAME_RESULT.shape == processed_frame.shape and
                PREVIOUS_FRAME_RESULT.dtype == processed_frame.dtype):
            try:
                final_frame = gpu_add_weighted(
                    PREVIOUS_FRAME_RESULT, 1.0 - interpolation_weight,
                    processed_frame, interpolation_weight, 0,
                )
                final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
            except cv2.error:
                final_frame = processed_frame
                PREVIOUS_FRAME_RESULT = None
        PREVIOUS_FRAME_RESULT = final_frame.copy()
    else:
        PREVIOUS_FRAME_RESULT = None

    return final_frame



def process_frame(source_face: Face, temp_frame: Frame, target_face: Face = None) -> Frame:
    """Process a single frame, swapping source_face onto detected target(s).

    Args:
        target_face: Pre-detected target face. When provided, skips the
            internal face detection call (saves ~30-40ms per frame).
            Ignored when many_faces mode is active.
    """
    if getattr(modules.globals, "opacity", 1.0) == 0:
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame

    processed_frame = temp_frame
    swapped_face_bboxes = []

    if modules.globals.many_faces:
        many_faces = get_many_faces(processed_frame)
        if many_faces:
            current_swap_target = processed_frame.copy()
            for face in many_faces:
                current_swap_target = swap_face(source_face, face, current_swap_target)
                if face is not None and hasattr(face, "bbox") and face.bbox is not None:
                    swapped_face_bboxes.append(face.bbox.astype(int))
            processed_frame = current_swap_target
    else:
        if target_face is None:
            target_face = get_one_face(processed_frame)
        if target_face:
            processed_frame = swap_face(source_face, target_face, processed_frame)
            if hasattr(target_face, "bbox") and target_face.bbox is not None:
                swapped_face_bboxes.append(target_face.bbox.astype(int))

    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)
    return final_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    """Handles complex mapping scenarios (map_faces=True) and live streams."""
    if getattr(modules.globals, "opacity", 1.0) == 0:
        # If opacity is 0, no swap happens, so no post-processing needed.
        # Also reset interpolation state if it was active.
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame

    processed_frame = temp_frame # Start with the input frame
    swapped_face_bboxes = [] # Keep track of where swaps happened

    # Determine source/target pairs based on mode
    source_target_pairs = []

    # Ensure maps exist before accessing them
    source_target_map = getattr(modules.globals, "source_target_map", None)
    simple_map = getattr(modules.globals, "simple_map", None)

    # Check if target is a file path (image or video) or live stream
    is_file_target = modules.globals.target_path and (is_image(modules.globals.target_path) or is_video(modules.globals.target_path))

    if is_file_target:
        # Processing specific image or video file with pre-analyzed maps
        if source_target_map:
            if modules.globals.many_faces:
                source_face = default_source_face() # Use default source for all targets
                if source_face:
                    for map_data in source_target_map:
                        if is_image(modules.globals.target_path):
                            target_info = map_data.get("target", {})
                            if target_info: # Check if target info exists
                                target_face = target_info.get("face")
                                if target_face:
                                    source_target_pairs.append((source_face, target_face))
                        elif is_video(modules.globals.target_path):
                             # Find faces for the current frame_path in video map
                             target_frames_data = map_data.get("target_faces_in_frame", [])
                             if target_frames_data: # Check if frame data exists
                                 target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                                 for frame_data in target_frames:
                                     faces_in_frame = frame_data.get("faces", [])
                                     if faces_in_frame: # Check if faces exist
                                         for target_face in faces_in_frame:
                                             source_target_pairs.append((source_face, target_face))
            else: # Single face or specific mapping
                 for map_data in source_target_map:
                    source_info = map_data.get("source", {})
                    if not source_info: continue # Skip if no source info
                    source_face = source_info.get("face")
                    if not source_face: continue # Skip if no source defined for this map entry

                    if is_image(modules.globals.target_path):
                        target_info = map_data.get("target", {})
                        if target_info:
                           target_face = target_info.get("face")
                           if target_face:
                              source_target_pairs.append((source_face, target_face))
                    elif is_video(modules.globals.target_path):
                        target_frames_data = map_data.get("target_faces_in_frame", [])
                        if target_frames_data:
                           target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                           for frame_data in target_frames:
                               faces_in_frame = frame_data.get("faces", [])
                               if faces_in_frame:
                                  for target_face in faces_in_frame:
                                      source_target_pairs.append((source_face, target_face))

    else:
        # Live stream or webcam processing (analyze faces on the fly)
        detected_faces = get_many_faces(processed_frame)
        if detected_faces:
            if modules.globals.many_faces:
                 source_face = default_source_face() # Use default source for all detected targets
                 if source_face:
                     for target_face in detected_faces:
                        source_target_pairs.append((source_face, target_face))
            elif simple_map:
                # Use simple_map (source_faces <-> target_embeddings)
                source_faces = simple_map.get("source_faces", [])
                target_embeddings = simple_map.get("target_embeddings", [])

                if source_faces and target_embeddings and len(source_faces) == len(target_embeddings):
                     # Match detected faces to the closest target embedding
                     if len(detected_faces) <= len(target_embeddings):
                          # More targets defined than detected - match each detected face
                          for detected_face in detected_faces:
                              if detected_face.normed_embedding is None: continue
                              closest_idx, _ = find_closest_centroid(target_embeddings, detected_face.normed_embedding)
                              if 0 <= closest_idx < len(source_faces):
                                  source_target_pairs.append((source_faces[closest_idx], detected_face))
                     else:
                          # More faces detected than targets defined - match each target embedding to closest detected face
                          detected_embeddings = [f.normed_embedding for f in detected_faces if f.normed_embedding is not None]
                          detected_faces_with_embedding = [f for f in detected_faces if f.normed_embedding is not None]
                          if not detected_embeddings: return processed_frame # No embeddings to match

                          for i, target_embedding in enumerate(target_embeddings):
                              if 0 <= i < len(source_faces): # Ensure source face exists for this embedding
                                 closest_idx, _ = find_closest_centroid(detected_embeddings, target_embedding)
                                 if 0 <= closest_idx < len(detected_faces_with_embedding):
                                     source_target_pairs.append((source_faces[i], detected_faces_with_embedding[closest_idx]))
            else: # Fallback: if no map, use default source for the single detected face (if any)
                source_face = default_source_face()
                target_face = get_one_face(processed_frame, detected_faces) # Use faces already detected
                if source_face and target_face:
                    source_target_pairs.append((source_face, target_face))


    # Perform swaps based on the collected pairs
    current_swap_target = processed_frame.copy() # Apply swaps sequentially
    for source_face, target_face in source_target_pairs:
        if source_face and target_face:
            current_swap_target = swap_face(source_face, target_face, current_swap_target)
            if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
                swapped_face_bboxes.append(target_face.bbox.astype(int))
    processed_frame = current_swap_target # Assign final result


    # Apply sharpening and interpolation
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)

    return final_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """
    Processes a list of frame paths (typically for video).
    Optimized with better memory management and caching.
    Iterates through frames, applies the appropriate swapping logic based on globals,
    and saves the result back to the frame path. Handles multi-threading via caller.
    """
    # Determine which processing function to use based on map_faces global setting
    use_v2 = getattr(modules.globals, "map_faces", False)
    source_face = None # Initialize source_face

    # --- Pre-load source face only if needed (Simple Mode: map_faces=False) ---
    if not use_v2:
        if not source_path or not os.path.exists(source_path):
            update_status(f"Error: Source path invalid or not provided for simple mode: {source_path}", NAME)
            # Log the error but allow proceeding; subsequent check will stop processing.
        else:
            try:
                source_img = cv2.imread(source_path)
                if source_img is None:
                    # Specific error for file reading failure
                    update_status("Error reading source image file. Please check file integrity.", NAME)
                    print(f"Path: {source_path}")
                else:
                    source_face = get_one_face(source_img)
                    if source_face is None:
                        # Specific message for no face detected after successful read
                        update_status("Warning: Successfully read source image, but no face detected.", NAME)
                        print(f"Path: {source_path}")
                    # Free memory immediately after extracting face
                    del source_img
            except Exception as e:
                # Print the specific exception caught
                import traceback
                print(f"{NAME}: Caught exception during source image processing for {source_path}:")
                traceback.print_exc() # Print the full traceback
                update_status(f"Error during source image reading or analysis {source_path}: {e}", NAME)
                # Log general exception during the process

    total_frames = len(temp_frame_paths)
    # update_status(f"Processing {total_frames} frames. Use V2 (map_faces): {use_v2}", NAME) # Optional Debug

    # --- Stop processing entirely if in Simple Mode and source face is invalid ---
    if not use_v2 and source_face is None:
        update_status(f"Halting video processing: Invalid or no face detected in source image for simple mode.", NAME)
        if progress:
            # Ensure the progress bar completes if it was started
            remaining_updates = total_frames - progress.n if hasattr(progress, 'n') else total_frames
            if remaining_updates > 0:
                progress.update(remaining_updates)
        return # Exit the function entirely

    # --- Process each frame path provided in the list ---
    # Note: In the current core.py multi_process_frame, temp_frame_paths will usually contain only ONE path per call.
    for i, temp_frame_path in enumerate(temp_frame_paths):
        # update_status(f"Processing frame {i+1}/{total_frames}: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug

        # Read the target frame
        temp_frame = None
        try:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                print(f"{NAME}: Error: Could not read frame: {temp_frame_path}, skipping.")
                if progress: progress.update(1)
                continue # Skip this frame if read fails
        except Exception as read_e:
            print(f"{NAME}: Error reading frame {temp_frame_path}: {read_e}, skipping.")
            if progress: progress.update(1)
            continue

        # Select processing function and execute
        result_frame = None
        try:
            if use_v2:
                # V2 uses global maps and needs the frame path for lookup in video mode
                # update_status(f"Using process_frame_v2 for: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug
                result_frame = process_frame_v2(temp_frame, temp_frame_path)
            else:
                # Simple mode uses the pre-loaded source_face (already checked for validity above)
                # update_status(f"Using process_frame (simple) for: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug
                result_frame = process_frame(source_face, temp_frame) # source_face is guaranteed to be valid here

            # Check if processing actually returned a frame
            if result_frame is None:
                 print(f"{NAME}: Warning: Processing returned None for frame {temp_frame_path}. Using original.")
                 result_frame = temp_frame

        except Exception as proc_e:
            print(f"{NAME}: Error processing frame {temp_frame_path}: {proc_e}")
            # import traceback # Optional for detailed debugging
            # traceback.print_exc()
            result_frame = temp_frame # Use original frame on processing error

        # Write the result back to the same frame path with optimized compression
        try:
            # Use PNG compression level 3 (faster) instead of default 9
            write_success = cv2.imwrite(temp_frame_path, result_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not write_success:
                print(f"{NAME}: Error: Failed to write processed frame to {temp_frame_path}")
        except Exception as write_e:
            print(f"{NAME}: Error writing frame {temp_frame_path}: {write_e}")
        
        # Free memory immediately after processing
        del temp_frame
        if result_frame is not None:
            del result_frame

        # Update progress bar
        if progress:
            progress.update(1)
        # else: # Basic console progress (optional)
        #     if (i + 1) % 10 == 0 or (i + 1) == total_frames: # Update every 10 frames or on last frame
        #        update_status(f"Processed frame {i+1}/{total_frames}", NAME)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Processes a single target image."""
    # --- Reset interpolation state for single image processing ---
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    # ---

    use_v2 = getattr(modules.globals, "map_faces", False)

    # Read target first
    try:
        target_frame = cv2.imread(target_path)
        if target_frame is None:
            update_status("Error: Could not read target image.", NAME)
            print(f"Path: {target_path}")
            return
    except Exception as read_e:
        update_status(f"Error reading target image {target_path}: {read_e}", NAME)
        return

    result = None
    try:
        if use_v2:
            if getattr(modules.globals, "many_faces", False):
                 update_status("Processing image with 'map_faces' and 'many_faces'. Using pre-analysis map.", NAME)
            # V2 processes based on global maps, doesn't need source_path here directly
            # Assumes maps are pre-populated. Pass target_path for map lookup.
            result = process_frame_v2(target_frame, target_path)

        else: # Simple mode
            try:
                source_img = cv2.imread(source_path)
                if source_img is None:
                    update_status(f"Error: Could not read source image: {source_path}", NAME)
                    return
                source_face = get_one_face(source_img)
                if not source_face:
                    update_status(f"Error: No face found in source image: {source_path}", NAME)
                    return
            except Exception as src_e:
                 update_status(f"Error reading or analyzing source image {source_path}: {src_e}", NAME)
                 return

            result = process_frame(source_face, target_frame)

        # Write the result if processing was successful
        if result is not None:
            ext = os.path.splitext(output_path)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                write_success = cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 97])
            elif ext == ".png":
                write_success = cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                write_success = cv2.imwrite(output_path, result)
            if write_success:
                update_status("Output image saved successfully.", NAME)
                print(f"Path: {output_path}")
            else:
                update_status("Error: Failed to write output image.", NAME)
                print(f"Path: {output_path}")
        else:
            # This case might occur if process_frame/v2 returns None unexpectedly
            update_status("Image processing failed (result was None).", NAME)

    except Exception as proc_e:
         update_status(f"Error during image processing: {proc_e}", NAME)
         # import traceback
         # traceback.print_exc()


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Sets up and calls the frame processing for video."""
    # --- Reset interpolation state before starting video processing ---
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    # ---

    mode_desc = "'map_faces'" if getattr(modules.globals, "map_faces", False) else "'simple'"
    if getattr(modules.globals, "map_faces", False) and getattr(modules.globals, "many_faces", False):
        mode_desc += " and 'many_faces'. Using pre-analysis map."
    update_status("Processing video...", NAME)
    print(f"Mode: {mode_desc}")

    # Pass the correct source_path (needed for simple mode in process_frames)
    # The core processing logic handles calling the right frame function (process_frames)
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames # Pass the newly modified process_frames
    )

# ==========================
# MASKING FUNCTIONS (Mostly unchanged, added safety checks and minor improvements)
# ==========================

def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None # Initialize
    mouth_box = (0,0,0,0) # Initialize

    # Validate face and landmarks
    if face is None or not hasattr(face, 'landmark_2d_106'):
        # print("Warning: Invalid face object passed to create_lower_mouth_mask.")
        return mask, mouth_cutout, mouth_box, lower_lip_polygon

    landmarks = face.landmark_2d_106

    # Check landmark validity
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        # print("Warning: Invalid or insufficient landmarks for mouth mask.")
        return mask, mouth_cutout, mouth_box, lower_lip_polygon

    try: # Wrap main logic in try-except
        # Use outer mouth landmarks (52-71) to capture the full mouth area
        # This covers both upper and lower lips for proper mouth preservation
        lower_lip_order = list(range(52, 72))

        # Check if all indices are valid for the loaded landmarks (already partially done by < 106 check)
        if max(lower_lip_order) >= landmarks.shape[0]:
            # print(f"Warning: Landmark index {max(lower_lip_order)} out of bounds for shape {landmarks.shape[0]}.")
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)

        # Filter out potential NaN or Inf values in landmarks
        if not np.all(np.isfinite(lower_lip_landmarks)):
            # print("Warning: Non-finite values detected in lower lip landmarks.")
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        center = np.mean(lower_lip_landmarks, axis=0)
        if not np.all(np.isfinite(center)): # Check center calculation
            # print("Warning: Could not calculate valid center for mouth mask.")
            return mask, mouth_cutout, mouth_box, lower_lip_polygon


        mouth_mask_size = getattr(modules.globals, "mouth_mask_size", 0.0) # 0-100 slider
        # 0=tight lip outline, 50=covers mouth area, 100=mouth to chin
        expansion_factor = 1 + (mouth_mask_size / 100.0) * 2.5

        # Expand landmarks from center, with extra downward bias toward chin
        offsets = lower_lip_landmarks - center
        # Add extra downward expansion for points below center (toward chin)
        chin_bias = 1 + (mouth_mask_size / 100.0) * 1.5  # extra vertical stretch downward
        scale_y = np.where(offsets[:, 1] > 0, expansion_factor * chin_bias, expansion_factor)
        expanded_landmarks = lower_lip_landmarks.copy()
        expanded_landmarks[:, 0] = center[0] + offsets[:, 0] * expansion_factor
        expanded_landmarks[:, 1] = center[1] + offsets[:, 1] * scale_y

        # Ensure landmarks are finite after adjustments
        if not np.all(np.isfinite(expanded_landmarks)):
            # print("Warning: Non-finite values detected after expanding landmarks.")
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        expanded_landmarks = expanded_landmarks.astype(np.int32)

        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Add padding *after* initial min/max calculation
        padding_ratio = 0.1 # Percentage padding
        padding_x = int((max_x - min_x) * padding_ratio)
        padding_y = int((max_y - min_y) * padding_ratio) # Use y-range for y-padding

        # Apply padding and clamp to frame boundaries
        frame_h, frame_w = frame.shape[:2]
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(frame_w, max_x + padding_x)
        max_y = min(frame_h, max_y + padding_y)


        if max_x > min_x and max_y > min_y:
            # Create the mask ROI
            mask_roi_h = max_y - min_y
            mask_roi_w = max_x - min_x
            mask_roi = np.zeros((mask_roi_h, mask_roi_w), dtype=np.uint8)

            # Shift polygon coordinates relative to the ROI's top-left corner
            polygon_relative_to_roi = expanded_landmarks - [min_x, min_y]

            # Draw polygon on the ROI mask
            cv2.fillPoly(mask_roi, [polygon_relative_to_roi], 255)

            # Apply Gaussian blur (GPU-accelerated when available)
            blur_k_size = getattr(modules.globals, "mask_blur_kernel", 15) # Default 15
            blur_k_size = max(1, blur_k_size // 2 * 2 + 1) # Ensure odd
            mask_roi = gpu_gaussian_blur(mask_roi, (blur_k_size, blur_k_size), 0)

            # Place the mask ROI in the full-sized mask
            mask[min_y:max_y, min_x:max_x] = mask_roi

            # Extract the masked area from the *original* frame
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

            lower_lip_polygon = expanded_landmarks # Return polygon in original frame coords
            mouth_box = (min_x, min_y, max_x, max_y) # Return the calculated box
        else:
            # print("Warning: Invalid mouth mask bounding box after padding/clamping.") # Optional debug
            pass

    except IndexError as idx_e:
        # print(f"Warning: Landmark index out of bounds during mouth mask creation: {idx_e}") # Optional debug
        pass
    except Exception as e:
        print(f"Error in create_lower_mouth_mask: {e}") # Print unexpected errors
        # import traceback
        # traceback.print_exc()
        pass

    # Return values, ensuring defaults if errors occurred
    return mask, mouth_cutout, mouth_box, lower_lip_polygon


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple
) -> Frame:

    # Validate inputs
    if frame is None or face is None or mouth_mask_data is None or len(mouth_mask_data) != 4:
        return frame # Return original frame if inputs are invalid

    mask, mouth_cutout, box, lower_lip_polygon = mouth_mask_data
    (min_x, min_y, max_x, max_y) = box

    # Check if polygon is valid for drawing
    if lower_lip_polygon is None or not isinstance(lower_lip_polygon, np.ndarray) or len(lower_lip_polygon) < 3:
        return frame # Cannot draw without a valid polygon

    vis_frame = frame.copy()
    height, width = vis_frame.shape[:2]

    # Ensure box coordinates are valid integers within frame bounds
    try:
        min_x, min_y = max(0, int(min_x)), max(0, int(min_y))
        max_x, max_y = min(width, int(max_x)), min(height, int(max_y))
    except ValueError:
        # print("Warning: Invalid coordinates for mask visualization box.")
        return frame

    if max_x <= min_x or max_y <= min_y:
        return frame # Invalid box

    # Draw the lower lip polygon (green outline)
    try:
         # Ensure polygon points are within frame boundaries before drawing
         safe_polygon = lower_lip_polygon.copy()
         safe_polygon[:, 0] = np.clip(safe_polygon[:, 0], 0, width - 1)
         safe_polygon[:, 1] = np.clip(safe_polygon[:, 1], 0, height - 1)
         cv2.polylines(vis_frame, [safe_polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    except Exception as e:
        print(f"Error drawing polygon for visualization: {e}") # Optional debug
        pass

    # Draw bounding box (red rectangle)
    cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

    # Optional: Add labels
    label_pos_y = min_y - 10 if min_y > 20 else max_y + 15 # Adjust position based on box location
    label_pos_x = min_x
    try:
        cv2.putText(vis_frame, "Mouth Mask", (label_pos_x, label_pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    except Exception as e:
        # print(f"Error drawing text for visualization: {e}") # Optional debug
        pass


    return vis_frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray, # Full face mask (for blending edges)
    mouth_polygon: np.ndarray, # Specific polygon for the mouth area itself
) -> np.ndarray:

    # Basic validation
    if (frame is None or mouth_cutout is None or mouth_box is None or
        face_mask is None or mouth_polygon is None):
        # print("Warning: Invalid input (None value) to apply_mouth_area") # Optional debug
        return frame
    if (mouth_cutout.size == 0 or face_mask.size == 0 or len(mouth_polygon) < 3):
        # print("Warning: Invalid input (empty array/polygon) to apply_mouth_area") # Optional debug
        return frame

    try: # Wrap main logic in try-except
        min_x, min_y, max_x, max_y = map(int, mouth_box) # Ensure integer coords
        box_width = max_x - min_x
        box_height = max_y - min_y

        # Check box validity
        if box_width <= 0 or box_height <= 0:
            # print("Warning: Invalid mouth box dimensions in apply_mouth_area.")
            return frame

        # Define the Region of Interest (ROI) on the target frame (swapped frame)
        frame_h, frame_w = frame.shape[:2]
        # Clamp coordinates strictly within frame boundaries
        min_y, max_y = max(0, min_y), min(frame_h, max_y)
        min_x, max_x = max(0, min_x), min(frame_w, max_x)

        # Recalculate box dimensions based on clamped coords
        box_width = max_x - min_x
        box_height = max_y - min_y
        if box_width <= 0 or box_height <= 0:
            # print("Warning: ROI became invalid after clamping in apply_mouth_area.")
            return frame # ROI is invalid

        roi = frame[min_y:max_y, min_x:max_x]

        # Ensure ROI extraction was successful
        if roi.size == 0:
            # print("Warning: Extracted ROI is empty in apply_mouth_area.")
            return frame

        # Resize mouth cutout from original frame to fit the ROI size
        resized_mouth_cutout = None
        if roi.shape[:2] != mouth_cutout.shape[:2]:
             # Check if mouth_cutout has valid dimensions before resizing
             if mouth_cutout.shape[0] > 0 and mouth_cutout.shape[1] > 0:
                  resized_mouth_cutout = gpu_resize(mouth_cutout, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
             else:
                 # print("Warning: mouth_cutout has invalid dimensions, cannot resize.")
                 return frame # Cannot proceed without valid cutout
        else:
             resized_mouth_cutout = mouth_cutout

        # If resize failed or original was invalid
        if resized_mouth_cutout is None or resized_mouth_cutout.size == 0:
            # print("Warning: Mouth cutout is invalid after resize attempt.")
            return frame

        # --- Mask Creation ---
        # Create a mask based on the mouth_polygon, relative to the ROI
        polygon_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask_roi, [adjusted_polygon.astype(np.int32)], 255)

        # Feather the edges with Gaussian blur for smooth blending
        feather_amount = max(1, min(30, min(box_width, box_height) // 8))
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(polygon_mask_roi.astype(np.float32), (kernel_size, kernel_size), 0)

        # Normalize to [0.0, 1.0]
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask = feathered_mask / max_val
        else:
            feathered_mask.fill(0.0)

        # --- Blending: paste original mouth onto swapped face ---
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            mask_3ch = feathered_mask[:, :, np.newaxis].astype(np.float32)
            inv_mask = 1.0 - mask_3ch

            # Blend: (original_mouth * mask) + (swapped_face * (1 - mask))
            blended_roi = (resized_mouth_cutout.astype(np.float32) * mask_3ch +
                           roi.astype(np.float32) * inv_mask)

            frame[min_y:max_y, min_x:max_x] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"Error applying mouth area: {e}") # Optional debug
        # import traceback
        # traceback.print_exc()
        pass # Don't crash, just return the frame as is

    return frame


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    """Creates a feathered mask covering the whole face area based on 106 landmarks.

    Improvements:
    - Uses the full jaw contour (lm 0-32) for a tighter jawline fit.
    - Forehead estimated from actual brow-to-chin distance.
    - Adaptive blur radius proportional to face size.
    """
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((0, 0), dtype=np.uint8)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if face is None or not hasattr(face, 'landmark_2d_106'):
        return mask

    landmarks = face.landmark_2d_106
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        return mask

    try:
        if not np.all(np.isfinite(landmarks)):
            return mask

        h, w = frame.shape[:2]
        lm = np.clip(landmarks, [0, 0], [w - 1, h - 1]).astype(np.int32)

        # Jaw contour: lm 0-32
        jaw = lm[0:33]

        # Forehead from brow midpoints pushed upward
        brows = lm[33:53]   # left brow 33-42, right brow 43-52
        chin  = lm[16].astype(np.float32)
        brow_center = brows.astype(np.float32).mean(axis=0)

        up = brow_center - chin
        up_len = float(np.linalg.norm(up))
        if up_len > 0:
            up_unit = up / up_len
            forehead = (brows.astype(np.float32) + up_unit * up_len * 0.75)
            forehead = np.clip(forehead, [0, 0], [w - 1, h - 1]).astype(np.int32)
            # Widen forehead by 10%
            fc = forehead.mean(axis=0)
            forehead = ((forehead.astype(np.float32) - fc) * 1.10 + fc).astype(np.int32)
            forehead = np.clip(forehead, [0, 0], [w - 1, h - 1])
            all_pts = np.concatenate([jaw, forehead], axis=0)
        else:
            all_pts = jaw

        hull = cv2.convexHull(all_pts.astype(np.float32))
        if hull is None or len(hull) < 3:
            return mask
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

        # Adaptive blur: 12% of face radius, minimum 7px
        face_area = float(np.sum(mask > 0))
        face_r    = max(1, int(np.sqrt(face_area / np.pi)))
        blur_r    = max(3, face_r // 8)
        k_blur    = 2 * blur_r + 1
        mask = cv2.GaussianBlur(mask, (k_blur, k_blur), 0)

    except Exception as e:
        print(f"Error creating face mask: {e}")

    return mask




def apply_color_transfer(source, target):
    """
    Apply color transfer using LAB color space. Handles potential division by zero and ensures output is uint8.
    """
    # Input validation
    if source is None or target is None or source.size == 0 or target.size == 0:
        # print("Warning: Invalid input to apply_color_transfer.")
        return source # Return original source if invalid input

    # Ensure images are 3-channel BGR uint8
    if len(source.shape) != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        # print("Warning: Source image for color transfer is not uint8 BGR.")
        # Attempt conversion if possible, otherwise return original
        try:
            if len(source.shape) == 2: # Grayscale
                source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            source = np.clip(source, 0, 255).astype(np.uint8)
            if len(source.shape)!= 3 or source.shape[2]!= 3: raise ValueError("Conversion failed")
        except Exception:
            return source
    if len(target.shape) != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
        # print("Warning: Target image for color transfer is not uint8 BGR.")
        try:
            if len(target.shape) == 2: # Grayscale
                target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
            target = np.clip(target, 0, 255).astype(np.uint8)
            if len(target.shape)!= 3 or target.shape[2]!= 3: raise ValueError("Conversion failed")
        except Exception:
             return source # Return original source if target invalid

    result_bgr = source # Default to original source in case of errors

    try:
        # Convert to float32 [0, 1] range for LAB conversion
        source_float = source.astype(np.float32) / 255.0
        target_float = target.astype(np.float32) / 255.0

        source_lab = cv2.cvtColor(source_float, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_float, cv2.COLOR_BGR2LAB)

        # Compute statistics
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)

        # Reshape for broadcasting
        source_mean = source_mean.reshape((1, 1, 3))
        source_std = source_std.reshape((1, 1, 3))
        target_mean = target_mean.reshape((1, 1, 3))
        target_std = target_std.reshape((1, 1, 3))

        # Avoid division by zero or very small std deviations (add epsilon)
        epsilon = 1e-6
        source_std = np.maximum(source_std, epsilon)
        # target_std = np.maximum(target_std, epsilon) # Target std can be small

        # Perform color transfer in LAB space
        result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean

        # --- No explicit clipping needed in LAB space typically ---
        # Clipping is handled implicitly by the conversion back to BGR and then to uint8

        # Convert back to BGR float [0, 1]
        result_bgr_float = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        # Clip final BGR values to [0, 1] range before scaling to [0, 255]
        result_bgr_float = np.clip(result_bgr_float, 0.0, 1.0)

        # Convert back to uint8 [0, 255]
        result_bgr = (result_bgr_float * 255.0).astype("uint8")

    except cv2.error as e:
         # print(f"OpenCV error during color transfer: {e}. Returning original source.") # Optional debug
         return source # Return original source if conversion fails
    except Exception as e:
         # print(f"Unexpected color transfer error: {e}. Returning original source.") # Optional debug
         # import traceback
         # traceback.print_exc()
         return source

    return result_bgr
