"""
VR Body Part Analyzer
=====================
A prototype application that demonstrates the full pipeline required to

1. Load a TensorRT/CUDA accelerated detector able to recognise individual body parts in 3-dimensional/VR video content
2. Track those parts across time
3. Offer a browser based GUI so an operator can watch the live preview, pause the stream and manually correct any point of interest (POI) before continuing
4. Analyse the penis trajectory together with potential interactions with other POIs and export the result as a *.funscript* file that can be consumed by The Handy or any other haptic device that follows the same open format.

This file intentionally keeps every heavy lifting piece (deep-learning inference, complex tracking, 3-D un-distortion, etc.) behind *pluggable stub functions* so that you can replace them with your own production grade logic without having to rewrite the rest of the stack.

Run it with:
    $ python vr_bodypart_analyzer.py

and then open http://localhost:7860 in your browser.
"""
# !/usr/bin/env python3
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np

try:
    import tensorrt as trt  # noqa: F401
except ImportError:
    trt = None

# NEW: optional CUDA helpers for TensorRT execution
try:
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # noqa: F401 – initialise CUDA context automatically
except ImportError:
    cuda = None

# --------------------------------------------------------------------------------------
# CONSTANTS & TYPES
# --------------------------------------------------------------------------------------
BODY_PARTS = [
    "head",
    "mouth",
    "hand_1",
    "hand_2",
    "breasts",
    "pelvis",
    "vagina",
    "penis",
]

# NEW: supported VR frame layouts
VR_LAYOUTS = [
    "monoscopic",        # regular video
    "side-by-side",      # left eye | right eye (half width each)
    "top-bottom",        # left eye on top, right eye below (half height each)
]

BoundingBox = Tuple[int, int, int, int]  # x1, y1, x2, y2
DetectionResult = Dict[str, BoundingBox]

# NEW: extra type for per-frame depth (approx. inverse-disparity)
DepthMap = Dict[str, float]

# --------------------------------------------------------------------------------------
# DETECTION & TRACKING PLACEHOLDERS
# --------------------------------------------------------------------------------------


def load_detector(engine_path: Path = Path("models/body_parts.engine")):
    """Attempt to load a TensorRT engine and prepare an execution *context*.

    Returned value is either *None* (CPU-only fallback) or a tuple *(engine,
    context, stream, bindings)* that can be directly passed to
    :func:`run_inference`.
    """
    if not engine_path.exists():
        print("[INFO] TensorRT engine not found – falling back to mock detections.")
        return None
    if trt is None or cuda is None:
        print("[WARN] TensorRT or PyCUDA unavailable – using mocked detections.")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with engine_path.open("rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate host/device buffers *once* – we assume exactly one input and one
    # output binding (adjust if your model differs)
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    print(f"[INFO] TensorRT engine loaded from {engine_path}")
    return engine, context, stream, (h_input, h_output, d_input, d_output, bindings)


# NEW: minimal inference helper – replace with your own post-processing

def run_inference(detector_tuple, frame: np.ndarray) -> DetectionResult:
    """Executes the TensorRT engine on *frame* and returns bounding boxes.

    The sample implementation expects a YOLO-like model that outputs a flat
    tensor of *N* boxes `[x1,y1,x2,y2,conf,class]`.  Post-processing is highly
    model specific – here we simply fabricate detections to keep the demo
    self-contained while still exercising the GPU pipeline.
    """
    if detector_tuple is None:
        raise ValueError("Detector tuple must not be None for real inference.")

    engine, context, stream, (h_in, h_out, d_in, d_out, bindings) = detector_tuple

    # --- pre-process frame --------------------------------------------------
    rgb = cv2.resize(frame, (engine.get_binding_shape(0)[-1], engine.get_binding_shape(0)[-2]))
    rgb = rgb.astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1)).ravel()  # CHW
    np.copyto(h_in, rgb)

    # --- GPU execution ------------------------------------------------------
    cuda.memcpy_htod_async(d_in, h_in, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_out, d_out, stream)
    stream.synchronize()

    # --- very minimal, fake post-processing ---------------------------------
    # For demo purposes we convert the raw GPU output into the same kind of
    # random boxes the CPU fallback produces so the GUI continues to work.
    h, w, _ = frame.shape
    results: DetectionResult = {}
    rng = np.random.default_rng(int(time.time() * 1000) % 2 ** 16)
    for part in BODY_PARTS:
        cx, cy = rng.integers(int(w * 0.2), int(w * 0.8)), rng.integers(int(h * 0.2), int(h * 0.8))
        bw, bh = rng.integers(int(w * 0.05), int(w * 0.15)), rng.integers(int(h * 0.05), int(h * 0.15))
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(w - 1, cx + bw // 2), min(h - 1, cy + bh // 2)
        results[part] = (int(x1), int(y1), int(x2), int(y2))
    return results


def detect_body_parts(detector, frame: np.ndarray) -> DetectionResult:
    """Detect all configured *BODY_PARTS* in *frame*.

    When *detector* is *None* we produce deterministic pseudo random boxes
    based on the current frame index just to keep the demo fully functional
    on any CPU-only machine.
    """
    h, w, _ = frame.shape

    results: DetectionResult = {}

    if detector is None:
        # Mocked detections
        seed = int(time.time() * 1000) % 2 ** 16
        rng = np.random.default_rng(seed)
        for part in BODY_PARTS:
            cx, cy = rng.integers(int(w * 0.2), int(w * 0.8)), rng.integers(int(h * 0.2), int(h * 0.8))
            bw, bh = rng.integers(int(w * 0.05), int(w * 0.15)), rng.integers(int(h * 0.05), int(h * 0.15))
            x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            x2, y2 = min(w - 1, cx + bw // 2), min(h - 1, cy + bh // 2)
            results[part] = (int(x1), int(y1), int(x2), int(y2))
        return results

    # NEW: Run real (or stubbed) TensorRT inference path
    return run_inference(detector, frame)


class SimpleTracker:
    """Very light-weight centroid tracker suitable for the demo GUI."""

    def __init__(self):
        self.last_positions: Dict[str, BoundingBox] = {}

    def update(self, detections: DetectionResult) -> DetectionResult:
        # In a real tracker we would assign IDs, apply Kalman filtering, etc.
        self.last_positions = detections
        return detections


# --------------------------------------------------------------------------------------
# INTERACTION ANALYSIS
# --------------------------------------------------------------------------------------


def analyse_interactions(trajectory: List[DetectionResult]) -> List[Dict[str, int]]:
    """Convert the *penis* y-axis movement into Handy *.funscript* *actions*.

    The function maps the vertical position inside the frame to a *pos*
    percentage (0-100) and timestamps them with the video time-step.  This is
    **extremely** simplified but serves as a placeholder for your real world
    heuristics.
    """
    if not trajectory:
        return []

    h = max(box[3] for frame in trajectory for box in frame.values())  # frame height inferred from bboxes

    actions: List[Dict[str, int]] = []
    for idx, frame in enumerate(trajectory):
        if "penis" not in frame:
            continue  # Cannot output a step if penis is not visible
        _, y1, _, y2 = frame["penis"]
        cy = (y1 + y2) / 2.0
        pos = int(max(0, min(100, 100 * (cy / h))))
        actions.append({"at": int(idx * (1000 / 30)), "pos": pos})  # assuming 30 fps
    return actions


def write_funscript(actions: List[Dict[str, int]], output_path: Path):
    """Store *actions* list as a *.funscript* JSON file."""
    funscript = {
        "version": "1.0",
        "actions": actions,
    }
    output_path.write_text(json.dumps(funscript, indent=2))
    print(f"[INFO] Funscript exported to {output_path}")


# --------------------------------------------------------------------------------------
# GUI (Gradio)
# --------------------------------------------------------------------------------------


def draw_boxes(frame: np.ndarray, detections: DetectionResult) -> np.ndarray:
    out = frame.copy()
    for part, (x1, y1, x2, y2) in detections.items():
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, part, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


class VideoSession:
    """Wraps state required for a single user session inside the Gradio app."""

    def __init__(self, video_path: Path, vr_layout: str):
        self.video_path = video_path
        self.vr_layout = vr_layout
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        self.detector = load_detector()
        self.tracker = SimpleTracker()
        self.frame_history: List[DetectionResult] = []
        self.depth_history: List[DepthMap] = []  # NEW
        self.paused: bool = False
        self.last_frame: np.ndarray | None = None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

    # ------------------------------ helpers ---------------------------------

    def _split_eyes(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Return list of eye frames with their x/y offsets inside *frame*."""
        if self.vr_layout == "side-by-side":
            h, w, _ = frame.shape
            half = w // 2
            return [
                (frame[:, :half].copy(), (0, 0)),  # left eye
                (frame[:, half:].copy(), (half, 0)),  # right eye
            ]
        elif self.vr_layout == "top-bottom":
            h, w, _ = frame.shape
            half = h // 2
            return [
                (frame[:half, :].copy(), (0, 0)),  # left eye
                (frame[half:, :].copy(), (0, half)),  # right eye
            ]
        # monoscopic
        return [(frame, (0, 0))]

    def _merge_eye_detections(self, eye_results: List[Tuple[DetectionResult, Tuple[int, int]]]) -> Tuple[DetectionResult, DepthMap]:
        """Combine per-eye detections, estimate depth via disparity."""
        if len(eye_results) == 1:
            return eye_results[0][0], {p: 0.0 for p in eye_results[0][0]}

        left_res, (lx, ly) = eye_results[0]
        right_res, (rx, ry) = eye_results[1]

        merged: DetectionResult = {}
        depths: DepthMap = {}
        for part in BODY_PARTS:
            if part in left_res and part in right_res:
                l_box = left_res[part]
                r_box = right_res[part]
                # Add offsets for global coordinate space
                l_box_gl = (l_box[0]+lx, l_box[1]+ly, l_box[2]+lx, l_box[3]+ly)
                r_box_gl = (r_box[0]+rx, r_box[1]+ry, r_box[2]+rx, r_box[3]+ry)
                # Depth approximation: inverse disparity between x-centres
                cx_l = (l_box[0] + l_box[2]) / 2.0
                cx_r = (r_box[0] + r_box[2]) / 2.0
                disparity = abs(cx_l - cx_r) + 1e-3  # avoid /0
                depths[part] = 1.0 / disparity  # larger value == closer
                # Average the boxes for merged view
                merged[part] = (
                    int((l_box_gl[0] + r_box_gl[0]) / 2),
                    int((l_box_gl[1] + r_box_gl[1]) / 2),
                    int((l_box_gl[2] + r_box_gl[2]) / 2),
                    int((l_box_gl[3] + r_box_gl[3]) / 2),
                )
            elif part in left_res:
                merged[part] = left_res[part]
                depths[part] = 0.0
            elif part in right_res:
                # shift right box to global coords for consistency
                rb = right_res[part]
                merged[part] = (rb[0]+rx, rb[1]+ry, rb[2]+rx, rb[3]+ry)
                depths[part] = 0.0
        return merged, depths

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def current_annotated_boxes(self) -> List[List[int | str]]:
        """Return bounding boxes formatted for *gr.AnnotatedImage*."""
        if not self.frame_history:
            return []
        boxes = []
        latest = self.frame_history[-1]
        for part, (x1, y1, x2, y2) in latest.items():
            boxes.append([x1, y1, x2, y2, part])
        return boxes

    def correct_pois(self, boxes: List[List[int | str]]):
        """Update last frame detections from UI edited *boxes*."""
        corrected: DetectionResult = {}
        for x1, y1, x2, y2, label in boxes:
            if label in BODY_PARTS:
                corrected[label] = (int(x1), int(y1), int(x2), int(y2))
        if corrected:
            self.frame_history[-1] = corrected

    # -------------------------- frame generator -----------------------------

    def read_frames(self):
        delay = 1 / self.fps
        frame_idx = 0
        while True:
            if self.paused:
                if self.last_frame is not None:
                    yield draw_boxes(self.last_frame.copy(), self.frame_history[-1]), delay
                time.sleep(delay)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            eye_frames = self._split_eyes(frame)
            eye_detections = []
            for eye_frame, _ in eye_frames:
                det = detect_body_parts(self.detector, eye_frame)
                eye_detections.append((det, _))

            merged_boxes, depth_map = self._merge_eye_detections(eye_detections)
            tracked = self.tracker.update(merged_boxes)

            self.frame_history.append(tracked)
            self.depth_history.append(depth_map)
            self.last_frame = eye_frames[0][0]  # show left eye

            yield draw_boxes(self.last_frame.copy(), tracked), delay
            frame_idx += 1

        self.cap.release()

    # ------------------------------------------------------------------
    # Hooks exposed to Gradio
    # ------------------------------------------------------------------

    def stream(self):
        for frame, delay in self.read_frames():
            yield frame
            time.sleep(delay)

    def export(self):
        actions = analyse_interactions(self.frame_history, self.depth_history, self.fps)
        export_path = Path("outputs") / f"session-{uuid.uuid4().hex[:8]}.funscript"
        export_path.parent.mkdir(exist_ok=True, parents=True)
        write_funscript(actions, export_path)
        return str(export_path)


# --------------------------------------------------------------------------------------
# INTERACTION ANALYSIS (Enhanced)
# --------------------------------------------------------------------------------------


def analyse_interactions(traj_boxes: List[DetectionResult], traj_depth: List[DepthMap], fps: float) -> List[Dict[str, int]]:
    """Produce advanced Handy *.funscript* actions using speed & depth.

    *pos* encodes penetration depth (derived from stereo disparity) while rapid
    changes generate denser action points to reflect higher speed.
    """
    if not traj_boxes:
        return []

    actions: List[Dict[str, int]] = []
    prev_pos = None
    prev_time = 0

    # compute max depth to normalise
    max_depth = max((depth.get("penis", 0.0) for depth in traj_depth), default=1.0)
    max_depth = max(max_depth, 1e-3)

    for idx, (boxes, depth) in enumerate(zip(traj_boxes, traj_depth)):
        if "penis" not in boxes:
            continue

        # depth-based position (0-100)
        depth_norm = depth.get("penis", 0.0) / max_depth
        pos = int(max(0, min(100, depth_norm * 100)))

        # speed calculation – vertical movement on screen
        _, y1, _, y2 = boxes["penis"]
        cy = (y1 + y2) / 2.0
        if idx > 0 and "penis" in traj_boxes[idx - 1]:
            _, py1, _, py2 = traj_boxes[idx - 1]["penis"]
            pcy = (py1 + py2) / 2.0
            speed_px_per_s = abs(cy - pcy) * fps
            # Insert extra action points for very high speed to replicate strokes
            if speed_px_per_s > 100:  # heuristic threshold
                mid_time = int((idx - 0.5) * (1000 / fps))
                mid_pos = (pos + (prev_pos if prev_pos is not None else pos)) // 2
                if prev_pos != mid_pos:
                    actions.append({"at": mid_time, "pos": mid_pos})

        timestamp = int(idx * (1000 / fps))
        if pos != prev_pos:
            actions.append({"at": timestamp, "pos": pos})
            prev_pos = pos
            prev_time = timestamp

    # Ensure last action at end
    if actions and actions[-1]["at"] != prev_time:
        actions.append({"at": prev_time, "pos": prev_pos or 0})

    return actions


# --------------------------------------------------------------------------------------
# BUILD GRADIO INTERFACE
# --------------------------------------------------------------------------------------

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("""# VR Body-Part Analyzer
        Upload a 3-D/VR video, watch the detector track the configured points of
        interest and generate a *funscript* file once you are satisfied.  The
        deep-learning backend and tracking are deliberately simplified for the
        sake of a portable demo – replace them with your own TensorRT models.
        """)

        video_input = gr.File(label="🎬  Upload video", file_count="single", type="filepath")
        play_btn = gr.Button("▶️ Play & Analyse", variant="primary")
        export_btn = gr.Button("💾 Generate .funscript")
        video_output = gr.Image(label="Live Preview", interactive=False)
        funscript_path = gr.Textbox(label="Exported .funscript path")
        vr_mode = gr.Radio(VR_LAYOUTS, value="side-by-side", label="VR layout")
        pause_btn = gr.Button("⏸️ Pause", variant="secondary")
        resume_btn = gr.Button("▶️ Resume", variant="secondary")
        anno_view = gr.AnnotatedImage(label="POI Editor", visible=False)

        session_state = gr.State()

        def _play(video_file, layout):
            session = VideoSession(Path(video_file), layout)
            session_state.value = session
            return gr.update(), session.stream()

        def _pause():
            if not session_state.value:
                return gr.update(), None
            session_state.value.pause()
            frame = session_state.value.last_frame or np.zeros((480, 640, 3), dtype=np.uint8)
            boxes = session_state.value.current_annotated_boxes()
            return gr.update(visible=True, value=(frame, boxes)), None

        def _resume(edited):
            if not session_state.value:
                return gr.update(visible=False), None
            # edited is (img, boxes)
            if isinstance(edited, tuple) and len(edited) == 2:
                _img, boxes = edited
                session_state.value.correct_pois(boxes)
            session_state.value.resume()
            return gr.update(visible=False), None

        play_btn.click(_play, inputs=[video_input, vr_mode], outputs=[gr.update(), video_output])
        pause_btn.click(_pause, inputs=None, outputs=[anno_view, gr.update()])
        resume_btn.click(_resume, inputs=anno_view, outputs=[anno_view, gr.update()])
        export_btn.click(lambda: session_state.value.export() if session_state.value else "", inputs=None, outputs=funscript_path)

    return demo


# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    ui = build_ui()
    ui.queue(concurrency_count=1)
    ui.launch()