s a static background, detects a target color in HSV space,
and replaces detected regions with the background to simulate invisibility.

Key Features:
-------------
- Robust background modeling with motion filtering
- Manual and automatic HSV color detection
- Adaptive mask correction
- Temporal mask stabilization
- Morphological mask refinement
- Alpha blending control
- Live telemetry (FPS, mask coverage, mode)

Controls:
---------
Q  : Quit
A  : Toggle auto HSV detection
B  : Re-capture background
[  : Decrease alpha blending
]  : Increase alpha blending
"""

import cv2
import numpy as np
import time
from collections import deque

# =============================================================================
#                               CONFIGURATION
# =============================================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)

BACKGROUND_FRAMES = 90
MOTION_THRESHOLD = 2.0

FPS_WINDOW = 20
MASK_HISTORY_SIZE = 5

MASK_KERNEL_SIZE = (5, 5)
GAUSSIAN_BLUR_SIZE = (7, 7)

SAMPLE_BOX_SIZE = 40

MIN_MASK_RATIO = 0.02
MAX_MASK_RATIO = 0.45

ALPHA_STEP = 0.05

# =============================================================================
#                               UI UTILITIES
# =============================================================================

def noop(_):
    """Dummy callback for trackbars."""
    pass


def setup_hsv_trackbars(window_name: str) -> None:
    """Create HSV control trackbars."""
    cv2.createTrackbar("LH", window_name, 60, 180, noop)
    cv2.createTrackbar("LS", window_name, 50, 255, noop)
    cv2.createTrackbar("LV", window_name, 40, 255, noop)
    cv2.createTrackbar("UH", window_name, 120, 180, noop)
    cv2.createTrackbar("US", window_name, 255, 255, noop)
    cv2.createTrackbar("UV", window_name, 255, 255, noop)


def read_hsv_trackbars(window_name: str):
    """Read HSV values from trackbars."""
    lower = np.array([
        cv2.getTrackbarPos("LH", window_name),
        cv2.getTrackbarPos("LS", window_name),
        cv2.getTrackbarPos("LV", window_name)
    ])
    upper = np.array([
        cv2.getTrackbarPos("UH", window_name),
        cv2.getTrackbarPos("US", window_name),
        cv2.getTrackbarPos("UV", window_name)
    ])
    return lower, upper

# =============================================================================
#                           FRAME PROCESSING
# =============================================================================

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Flip and resize frame."""
    frame = cv2.flip(frame, 1)
    return cv2.resize(frame, FRAME_SIZE)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Apply morphological operations and blur to clean the mask."""
    kernel = np.ones(MASK_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, GAUSSIAN_BLUR_SIZE, 0)
    return mask


def stabilize_mask(mask: np.ndarray, history: deque) -> np.ndarray:
    """Smooth mask across time to reduce flicker."""
    history.append(mask)
    return np.mean(history, axis=0).astype(np.uint8)


def compute_mask_ratio(mask: np.ndarray) -> float:
    """Return ratio of active pixels in the mask."""
    return np.count_nonzero(mask) / mask.size

# =============================================================================
#                           BACKGROUND MODEL
# =============================================================================

def capture_background(cap) -> np.ndarray | None:
    """
    Capture a clean static background using motion filtering.
    Frames with noticeable motion are ignored.
    """
    frames = []
    prev_frame = None

    print("Capturing background... Please remain still.")

    for _ in range(BACKGROUND_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = preprocess_frame(frame)

        if prev_frame is not None:
            diff = np.mean(cv2.absdiff(frame, prev_frame))
            if diff > MOTION_THRESHOLD:
                continue

        prev_frame = frame.copy()
        frames.append(frame)

        cv2.imshow("Background Capture", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Background Capture")

    if not frames:
        return None

    return np.median(frames, axis=0).astype(np.uint8)

# =============================================================================
#                       AUTO HSV COLOR DETECTION
# =============================================================================

def estimate_hsv_from_center(hsv: np.ndarray):
    """
    Estimate HSV range by sampling a square region at the center of the frame.
    """
    h, w, _ = hsv.shape
    r = SAMPLE_BOX_SIZE

    region = hsv[h//2 - r:h//2 + r, w//2 - r:w//2 + r]
    median = np.median(region.reshape(-1, 3), axis=0)

    lower = np.array([max(0, median[0] - 18), 40, 40])
    upper = np.array([min(180, median[0] + 18), 255, 255])

    return lower, upper

# =============================================================================
#                           TELEMETRY OVERLAY
# =============================================================================

def draw_telemetry(
    frame: np.ndarray,
    fps_log: deque,
    mask_ratio: float,
    alpha: float,
    auto_mode: bool
) -> None:
    """Render system telemetry on the output frame."""
    avg_fps = sum(fps_log) / len(fps_log)

    lines = [
        f"FPS: {avg_fps:.1f}",
        f"Mask Coverage: {mask_ratio * 100:.1f}%",
        f"Alpha: {alpha:.2f}",
        f"Mode: {'AUTO' if auto_mode else 'MANUAL'}"
    ]

    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (10, 30 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# =============================================================================
#                           MAIN APPLICATION
# =============================================================================

def invisibility_cloak() -> None:
    """Run the invisibility cloak system."""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera access failed.")
        return

    control_window = "HSV Controls"
    cv2.namedWindow(control_window)
    setup_hsv_trackbars(control_window)

    background = capture_background(cap)
    if background is None:
        print("Background capture failed.")
        return

    fps_log = deque(maxlen=FPS_WINDOW)
    mask_history = deque(maxlen=MASK_HISTORY_SIZE)

    auto_mode = False
    alpha = 1.0
    last_valid_hsv = None
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = (
            estimate_hsv_from_center(hsv)
            if auto_mode
            else read_hsv_trackbars(control_window)
        )

        mask = cv2.inRange(hsv, lower, upper)
        mask = refine_mask(mask)

        ratio = compute_mask_ratio(mask)

        # Adaptive correction
        if ratio < MIN_MASK_RATIO and last_valid_hsv:
            lower, upper = last_valid_hsv
            mask = cv2.inRange(hsv, lower, upper)

        elif ratio > MAX_MASK_RATIO:
            lower[1] = min(255, lower[1] + 10)
            mask = cv2.inRange(hsv, lower, upper)

        if MIN_MASK_RATIO < ratio < MAX_MASK_RATIO:
            last_valid_hsv = (lower.copy(), upper.copy())

        mask = stabilize_mask(mask, mask_history)
        inverse_mask = cv2.bitwise_not(mask)

        cloak_region = cv2.bitwise_and(background, background, mask=mask)
        visible_region = cv2.bitwise_and(frame, frame, mask=inverse_mask)

        output = cv2.addWeighted(
            cloak_region, alpha,
            visible_region, 1 - alpha,
            0
        )

        # FPS calculation
        current_time = time.time()
        fps = 1 / max(current_time - prev_time, 1e-6)
        prev_time = current_time
        fps_log.append(fps)

        draw_telemetry(output, fps_log, ratio, alpha, auto_mode)
        cv2.imshow("Invisibility Cloak", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_mode = not auto_mode
        elif key == ord('b'):
            background = capture_background(cap)
        elif key == ord('['):
            alpha = max(0.0, alpha - ALPHA_STEP)
        elif key == ord(']'):
            alpha = min(1.0, alpha + ALPHA_STEP)

    cap.release()
    cv2.destroyAllWindows()

# =============================================================================
#                               ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    invisibility_cloak()

