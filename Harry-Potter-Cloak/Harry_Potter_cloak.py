ort numpy as np
import time
import os
from collections import deque

# ============================================================
#                    GLOBAL CONFIGURATION
# ============================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

BACKGROUND_FRAMES = 80
MOTION_THRESHOLD = 2.0
STABILITY_WINDOW = 15

MASK_KERNEL_SIZE = (5, 5)
GAUSSIAN_BLUR_SIZE = (7, 7)
SHARPEN_KERNEL = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

MASK_PREVIEW = True
MASK_BLEND = 1.0

OUTLINE_COLOR = (0, 255, 255)
OUTLINE_THICKNESS = 2

SAMPLE_BOX_SIZE = 40
AUTO_RECALIBRATE_INTERVAL = 6

CALIBRATION_FILE = "hsv_profile.npy"

BACKGROUND_EFFECTS = ["normal", "blur", "pixelate", "freeze"]
current_effect = 0


# ============================================================
#                    TRACKBAR SYSTEM
# ============================================================

def nothing(_):
    pass


def setup_trackbars(window_name):
    cv2.createTrackbar("LH", window_name, 60, 180, nothing)
    cv2.createTrackbar("LS", window_name, 50, 255, nothing)
    cv2.createTrackbar("LV", window_name, 40, 255, nothing)
    cv2.createTrackbar("UH", window_name, 120, 180, nothing)
    cv2.createTrackbar("US", window_name, 255, 255, nothing)
    cv2.createTrackbar("UV", window_name, 255, 255, nothing)


def get_trackbar_values(window_name):
    lower = np.array([
        cv2.getTrackbarPos("LH", window_name),
        cv2.getTrackbarPos("LS", window_name),
        cv2.getTrackbarPos("LV", window_name),
    ])

    upper = np.array([
        cv2.getTrackbarPos("UH", window_name),
        cv2.getTrackbarPos("US", window_name),
        cv2.getTrackbarPos("UV", window_name),
    ])
    return lower, upper


# ============================================================
#                    BACKGROUND CAPTURE
# ============================================================

def capture_background(cap):
    print("\nCapturing background. Please move out of frame.")

    frames = []
    prev = None

    for _ in range(BACKGROUND_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if prev is not None:
            diff = cv2.absdiff(frame, prev)
            if np.mean(diff) > MOTION_THRESHOLD:
                continue

        prev = frame.copy()
        frames.append(frame)

        cv2.imshow("Background Capture", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Background Capture")

    if len(frames) == 0:
        print("Background capture failed.")
        return None

    print("Background captured successfully.")
    return np.median(frames, axis=0).astype(np.uint8)


# ============================================================
#                       BACKGROUND EFFECTS
# ============================================================

def apply_background_effect(bg, effect):
    if effect == "normal":
        return bg

    if effect == "blur":
        return cv2.GaussianBlur(bg, (21, 21), 0)

    if effect == "pixelate":
        small = cv2.resize(bg, (64, 48))
        return cv2.resize(small, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)

    if effect == "freeze":
        return bg.copy()

    return bg


# ============================================================
#                    MASK PROCESSING
# ============================================================

def process_mask(mask):
    kernel = np.ones(MASK_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, GAUSSIAN_BLUR_SIZE, 0)
    return mask


def draw_outline(frame, mask):
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, OUTLINE_COLOR, OUTLINE_THICKNESS)
    return frame


# ============================================================
#                  COLOR AUTO DETECTION
# ============================================================

def get_auto_color(hsv):
    h, w, _ = hsv.shape
    r = SAMPLE_BOX_SIZE

    region = hsv[h//2-r:h//2+r, w//2-r:w//2+r]
    avg = np.median(region.reshape(-1, 3), axis=0)

    lower = np.array([max(0, avg[0] - 20), 40, 40])
    upper = np.array([min(180, avg[0] + 20), 255, 255])
    return lower, upper


# ============================================================
#                SAVE + LOAD HSV CALIBRATION
# ============================================================

def save_hsv_profile(lower, upper):
    np.save(CALIBRATION_FILE, np.array([lower, upper]))
    print("HSV calibration saved.")


def load_hsv_profile():
    if not os.path.exists(CALIBRATION_FILE):
        return None, None

    data = np.load(CALIBRATION_FILE)
    print("HSV calibration loaded from file.")
    return data[0], data[1]


# ============================================================
#                        MAIN LOOP
# ============================================================

def invisibility_cloak():

    global current_effect

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not detected.")
        return

    win = "Cloak Controls"
    cv2.namedWindow(win)
    setup_trackbars(win)

    loaded_lower, loaded_upper = load_hsv_profile()

    background = capture_background(cap)
    if background is None:
        return

    auto_detect = False
    use_saved_profile = loaded_lower is not None

    fps_history = deque(maxlen=STABILITY_WINDOW)
    prev = time.time()
    last_auto_recalibrate = time.time()

    snapshot_dir = "Snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("\n=== Keyboard Commands ===")
    print("q = Quit")
    print("a = Toggle Auto Color Detection")
    print("b = Re-Capture Background")
    print("e = Cycle Background Effect (blur, pixelate, freeze)")
    print("p = Save HSV Calibration")
    print("l = Load HSV Calibration")
    print("s = Save Snapshot")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Choose HSV range
        if auto_detect:
            lower, upper = get_auto_color(hsv)
        elif use_saved_profile:
            lower, upper = loaded_lower, loaded_upper
        else:
            lower, upper = get_trackbar_values(win)

        # Mask and effect
        mask = cv2.inRange(hsv, lower, upper)
        mask = process_mask(mask)

        inv = cv2.bitwise_not(mask)

        # Enhanced background
        bg_effect = apply_background_effect(background, BACKGROUND_EFFECTS[current_effect])

        cloak = cv2.bitwise_and(bg_effect, bg_effect, mask=mask)
        visible = cv2.bitwise_and(frame, frame, mask=inv)

        final = cv2.add(cloak, visible)
        final = draw_outline(final, mask)

        # FPS calculation
        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(final, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)

        mode = "AUTO" if auto_detect else "SAVED" if use_saved_profile else "MANUAL"
        cv2.putText(final, f"Mode: {mode}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

        cv2.putText(final, f"Background Effect: {BACKGROUND_EFFECTS[current_effect]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)

        cv2.imshow("Invisibility Cloak", final)

        if MASK_PREVIEW:
            cv2.imshow("Mask", mask)
            cv2.imshow("Mask Inv", inv)

        # Key Bindings
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("a"):
            auto_detect = not auto_detect
        elif key == ord("b"):
            background = capture_background(cap)
        elif key == ord("e"):
            current_effect = (current_effect + 1) % len(BACKGROUND_EFFECTS)
        elif key == ord("p"):
            save_hsv_profile(lower, upper)
        elif key == ord("l"):
            loaded_lower, loaded_upper = load_hsv_profile()
            use_saved_profile = True
        elif key == ord("s"):
            path = os.path.join(snapshot_dir, f"cloak_{int(time.time())}.png")
            cv2.imwrite(path, final)
            print("Saved:", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    invisibility_cloak()
