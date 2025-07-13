import numpy as np
import pytest

from vr_bodypart_analyzer import VideoSession, BODY_PARTS


def dummy_session(layout="side-by-side"):
    vs = VideoSession.__new__(VideoSession)  # bypass __init__
    vs.vr_layout = layout
    return vs


def test_split_eyes_side_by_side():
    vs = dummy_session("side-by-side")
    frame = np.zeros((480, 960, 3), dtype=np.uint8)
    eyes = vs._split_eyes(frame)
    assert len(eyes) == 2
    assert eyes[0][0].shape[1] == 480  # half width
    assert eyes[1][0].shape[1] == 480


def test_split_eyes_top_bottom():
    vs = dummy_session("top-bottom")
    frame = np.zeros((480, 960, 3), dtype=np.uint8)
    eyes = vs._split_eyes(frame)
    assert len(eyes) == 2
    assert eyes[0][0].shape[0] == 240  # half height
    assert eyes[1][0].shape[0] == 240


def test_merge_detection_depth():
    vs = dummy_session("side-by-side")
    left = {"penis": (10, 10, 30, 30)}
    right = {"penis": (50, 10, 70, 30)}
    merged, depth = vs._merge_eye_detections([(left, (0, 0)), (right, (480, 0))])
    assert "penis" in merged
    assert depth["penis"] > 0