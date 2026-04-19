"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation, pre/post processing, and the core
enhance-face-via-ONNX pipeline.
"""

import os
import platform
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime

import modules.globals

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
# On Intel Mac, CoreML is ineffective — force CPU-only for consistent performance
IS_INTEL_MAC = platform.system() == "Darwin" and platform.machine() == "x86_64"

# Intel i5-7267U has 2 physical / 4 logical cores.
# intra_op uses all 4 logical threads per ONNX session.
# We allow only 1 ONNX call at a time (semaphore=1) so two concurrent
# workers don't both saturate all 4 HT threads simultaneously.
_CPU_COUNT = max(1, os.cpu_count() or 1)  # 4 on Intel Core i5 dual-core
# Semaphore = 1: one ONNX inference at a time (avoids HT over-subscription)
# when execution_threads=2 workers are active
THREAD_SEMAPHORE = threading.Semaphore(1)

# Cache for session input name (to avoid per-frame metadata calls)
_SESSION_INPUT_NAME_CACHE: dict = {}


def build_provider_config(providers=None):
    """Wrap raw provider name strings with optimised CUDA / CoreML options.

    On Intel Mac (x86_64) CoreMLExecutionProvider is not effective — it
    falls back to CPU anyway and adds overhead.  We strip it and use only
    CPUExecutionProvider with maximum thread utilisation.
    """
    if providers is None:
        providers = modules.globals.execution_providers

    # Intel Mac: force pure CPU for predictable, optimal performance
    if IS_INTEL_MAC:
        return ['CPUExecutionProvider']

    config = []
    for p in providers:
        if isinstance(p, tuple):
            config.append(p)
        elif p == "CUDAExecutionProvider":
            config.append((
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "cudnn_conv_use_max_workspace": "1",
                    "do_copy_in_default_stream": "0",
                },
            ))
        elif p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                },
            ))
        else:
            config.append(p)
    return config


def run_inference(session: onnxruntime.InferenceSession,
                  input_name: str,
                  input_tensor: "np.ndarray") -> "np.ndarray":
    """Run ONNX inference, using IO binding when a CUDA session is active.

    IO binding avoids redundant host↔device copies by transferring the
    input tensor directly to GPU memory and letting ONNX Runtime allocate
    the output on the device.  Falls back to the standard ``session.run``
    path for non-CUDA providers or if binding fails.
    """
    if "CUDAExecutionProvider" in session.get_providers():
        try:
            io_binding = session.io_binding()

            # Input: numpy → GPU
            ort_input = onnxruntime.OrtValue.ortvalue_from_numpy(
                input_tensor, "cuda", 0,
            )
            io_binding.bind_ortvalue_input(input_name, ort_input)

            # Output: allocate on GPU (avoids a CPU-side allocation)
            output_name = session.get_outputs()[0].name
            io_binding.bind_output(output_name, "cuda", 0)

            session.run_with_iobinding(io_binding)

            return io_binding.get_outputs()[0].numpy()
        except Exception:
            # Fall back to standard path (e.g. ORT version mismatch,
            # unsupported op, or VRAM pressure)
            pass

    return session.run(None, {input_name: input_tensor})[0]


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session optimised for Intel Core i5-7267U (2P/4HT cores)."""
    providers = build_provider_config()
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    # Use all 4 logical threads for intra-op (single ONNX session runs fastest
    # by filling all HT threads with the one active inference).
    session_options.intra_op_num_threads = _CPU_COUNT      # 4 logical
    # inter_op = 1: we only allow 1 ONNX call at a time (THREAD_SEMAPHORE=1),
    # so spawning parallel inter-op threads just adds overhead.
    session_options.inter_op_num_threads = 1
    session_options.enable_mem_pattern = True
    session_options.enable_mem_reuse = True
    # Disable spin-wait on Intel to avoid burning all 4 threads in idle
    session_options.add_session_config_entry(
        'session.intra_op.allow_spinning', '0'
    )
    # Disable spinning on inter-op as well
    session_options.add_session_config_entry(
        'session.inter_op.allow_spinning', '0'
    )
    session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=providers,
    )
    warmup_session(session)
    return session


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / compile caching."""
    try:
        input_feed = {
            inp.name: np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
                dtype=np.float32,
            )
            for inp in session.get_inputs()
        }
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    Uses LANCZOS4 for high-quality downsampling (preserves fine hair/skin detail).
    """
    # LANCZOS4 gives sharper, more accurate texture than INTER_LINEAR
    resized = cv2.resize(face_img, (input_size, input_size),
                         interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def postprocess_face(output: np.ndarray, sharpen: bool = True) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8.

    Optionally applies a mild unsharp mask to recover fine detail
    that gets slightly smoothed by the GPEN restoration network.
    """
    img = output[0].transpose(1, 2, 0)
    img = ((img + 1.0) / 2.0 * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if sharpen:
        # Mild unsharp mask: recover detail lost in GPEN's restoration
        # sigma=1.0 targets fine skin texture; amount=0.45 is subtle
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = cv2.addWeighted(img, 1.45, blur, -0.45, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _get_face_affine(face: Any, input_size: int):
    """Compute affine transform to align a face to GPEN input space.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    template = np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model."""
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )

    blob = preprocess_face(face_crop, input_size)
    with THREAD_SEMAPHORE:
        # Cache input name to avoid repeated metadata lookups per frame
        session_id = id(session)
        if session_id not in _SESSION_INPUT_NAME_CACHE:
            _SESSION_INPUT_NAME_CACHE[session_id] = session.get_inputs()[0].name
        input_name = _SESSION_INPUT_NAME_CACHE[session_id]
        output = run_inference(session, input_name, blob)
    enhanced = postprocess_face(output, sharpen=True)

    # Feathered blend mask — wider border (1/12) for smoother edge
    mask = np.ones((input_size, input_size), dtype=np.float32)
    border = max(2, input_size // 12)
    mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask[:, :border] = np.minimum(mask[:, :border],
                                   np.linspace(0, 1, border)[np.newaxis, :])
    mask[:, -border:] = np.minimum(mask[:, -border:],
                                    np.linspace(1, 0, border)[np.newaxis, :])

    h, w = frame.shape[:2]
    # LANCZOS4 for warp-back: avoids ringing at high-contrast edges
    warped_enhanced = cv2.warpAffine(
        enhanced, inv_M, (w, h),
        flags=cv2.INTER_LANCZOS4, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    mask_3ch = warped_mask[:, :, np.newaxis]
    result = (warped_enhanced.astype(np.float32) * mask_3ch +
              frame.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)
