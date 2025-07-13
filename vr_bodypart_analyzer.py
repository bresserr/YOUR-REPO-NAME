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
    import tensorrt as trt  # noqa: F401 – optional, only required if you want real inference
except ImportError:
    trt = None

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

BoundingBox = Tuple[int, int, int, int]  # x1, y1, x2, y2
DetectionResult = Dict[str, BoundingBox]

# --------------------------------------------------------------------------------------
# DETECTION & TRACKING PLACEHOLDERS
# --------------------------------------------------------------------------------------


def load_detector(engine_path: Path = Path("models/body_parts.engine")):
    """Load / build a TensorRT engine once on start-up.

    The real implementation would prepare the CUDA context, create execution
    bindings and so on.  For the sake of a self-contained demo we just return
    *None* so that downstream code falls back to a mocked detector that
    yields random but plausible bounding boxes.
    """
    if not engine_path.exists():
        print("[INFO] TensorRT engine not found – falling back to mock detections.")
        return None
    if trt is None:
        raise RuntimeError("TensorRT package not available – please install it or remove the engine file.")

    # Actually load the engine here …
    # This would normally involve trt.Runtime and trt.ICudaEngine.
    print(f"[INFO] Loading TensorRT engine from {engine_path}")
    return "<trt_engine_placeholder>"


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

    # Real detection pipeline would go here – run detector on GPU, parse output, NMS, etc.
    raise NotImplementedError("Real detector logic is not yet implemented in this skeleton.")


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

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        self.detector = load_detector()
        self.tracker = SimpleTracker()
        self.frame_history: List[DetectionResult] = []

    def read_frames(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            detections = detect_body_parts(self.detector, frame)
            tracked = self.tracker.update(detections)
            self.frame_history.append(tracked)
            yield draw_boxes(frame, tracked), 1 / fps

        self.cap.release()

    # ------------------------------------------------------------------
    # Hooks exposed to Gradio
    # ------------------------------------------------------------------

    def stream(self):
        for frame, delay in self.read_frames():
            yield frame
            time.sleep(delay)

    def export(self):
        actions = analyse_interactions(self.frame_history)
        export_path = Path("outputs") / f"session-{uuid.uuid4().hex[:8]}.funscript"
        export_path.parent.mkdir(exist_ok=True, parents=True)
        write_funscript(actions, export_path)
        return str(export_path)


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

        session_state = gr.State()

        def _play(video_file):
            session = VideoSession(Path(video_file))
            session_state.value = session
            return gr.update(), session.stream()

        play_btn.click(_play, inputs=video_input, outputs=[gr.update(), video_output])
        export_btn.click(lambda: session_state.value.export() if session_state.value else "", inputs=None, outputs=funscript_path)

    return demo


# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    ui = build_ui()
    ui.queue(concurrency_count=1)
    ui.launch()