import cv2
import numpy as np
import mediapipe as mp
import time
from ai_features import ShapeRecognizer, auto_correct_shape, StrokeSmoother, DrawingStats

# =============================================================================
# EMA Smoothing for Landmark Coordinates
# =============================================================================
class SmoothLandmark:
    """Exponential Moving Average filter for hand landmark coordinates."""

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.prev = None

    def update(self, point):
        """Update with a new (x, y) point and return smoothed result."""
        if self.prev is None:
            self.prev = point
            return point
        sx = int(self.alpha * point[0] + (1 - self.alpha) * self.prev[0])
        sy = int(self.alpha * point[1] + (1 - self.alpha) * self.prev[1])
        self.prev = (sx, sy)
        return (sx, sy)

# =============================================================================
# Multi-Finger Gesture Detection
# =============================================================================
# MediaPipe hand landmark indices
# Thumb: 4 (tip), 3 (ip), 2 (mcp)
# Index: 8 (tip), 6 (pip)
# Middle: 12 (tip), 10 (pip)
# Ring: 16 (tip), 14 (pip)
# Pinky: 20 (tip), 18 (pip)

FINGER_TIP_IDS = [4, 8, 12, 16, 20]
FINGER_PIP_IDS = [3, 6, 10, 14, 18]  # For thumb, using IP joint

def count_fingers_up(lm_list):
    """
    Count how many fingers are raised.
    Returns a list of 5 booleans [thumb, index, middle, ring, pinky].
    """
    fingers = []

    # Thumb — compare x-coordinates (works for right hand in mirrored frame)
    if lm_list[4][0] < lm_list[3][0]:
        fingers.append(True)
    else:
        fingers.append(False)

    # Other 4 fingers — tip above pip means finger is up
    for tip_id, pip_id in zip(FINGER_TIP_IDS[1:], FINGER_PIP_IDS[1:]):
        if lm_list[tip_id][1] < lm_list[pip_id][1]:
            fingers.append(True)
        else:
            fingers.append(False)

    return fingers

def get_gesture(fingers):
    """
    Determine gesture from finger states.
    Returns: 'DRAW', 'SELECT', 'UNDO', 'REDO', or 'IDLE'
    """
    up_count = sum(fingers)

    # Index only up → DRAW mode
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return 'DRAW'

    # Index + Middle up (peace sign) → SELECT mode
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return 'SELECT'

    # Fist (no fingers up) → UNDO
    if up_count == 0:
        return 'UNDO'

    # All fingers up (open palm) → REDO
    if up_count == 5:
        return 'REDO'

    return 'IDLE'

# =============================================================================
# MediaPipe Setup — New Tasks API (MediaPipe 1.0+)
# =============================================================================
import os

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks
DrawingSpec = mp.tasks.vision.drawing_utils.DrawingSpec

# Path to the hand landmarker model (in same directory as this script)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8,
    running_mode=RunningMode.VIDEO,
)

landmarker = HandLandmarker.create_from_options(options)

# Custom drawing styles
landmark_style = DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=2)
connection_style = DrawingSpec(color=(0, 200, 180), thickness=1)

# =============================================================================
# Webcam
# =============================================================================
cap = cv2.VideoCapture(0)

screen_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =============================================================================
# Window
# =============================================================================
WINDOW_NAME = "Virtual Drawing — AI Enhanced"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# =============================================================================
# Canvas & Undo/Redo Stack
# =============================================================================
canvas = np.zeros((screen_h, screen_w, 3), np.uint8)

MAX_HISTORY = 20
undo_stack = []
redo_stack = []

def save_state():
    """Save current canvas state to undo stack."""
    global undo_stack, redo_stack
    undo_stack.append(canvas.copy())
    if len(undo_stack) > MAX_HISTORY:
        undo_stack.pop(0)
    redo_stack.clear()

def undo():
    """Undo last action."""
    global canvas, undo_stack, redo_stack
    if undo_stack:
        redo_stack.append(canvas.copy())
        canvas = undo_stack.pop()

def redo():
    """Redo last undone action."""
    global canvas, undo_stack, redo_stack
    if redo_stack:
        undo_stack.append(canvas.copy())
        canvas = redo_stack.pop()

# =============================================================================
# Colors (BGR)
# =============================================================================
BLUE   = (255, 50, 50)
GREEN  = (50, 220, 50)
RED    = (50, 50, 255)
YELLOW = (0, 230, 255)
CYAN   = (255, 220, 0)
PURPLE = (200, 50, 200)
BLACK  = (0, 0, 0)

draw_color = BLUE
brush_thickness = 7
eraser_thickness = 50

xp, yp = 0, 0

# =============================================================================
# Toolbar Layout
# =============================================================================
BUTTON_W = 100
BUTTON_H = 45
GAP = 12

TOOLS = [
    ["BLUE", "GREEN", "RED", "YELLOW", "CYAN", "PURPLE"],
    ["ERASE", "CLEAR", "UNDO", "REDO", "SHAPES", "SAVE", "EXIT"]
]

COLORS = {
    "BLUE": BLUE,
    "GREEN": GREEN,
    "RED": RED,
    "YELLOW": YELLOW,
    "CYAN": CYAN,
    "PURPLE": PURPLE,
}

start_x_row = []
for row in TOOLS:
    total_w = len(row) * BUTTON_W + (len(row) - 1) * GAP
    start_x_row.append((screen_w - total_w) // 2)

row1_y1 = 8
row1_y2 = row1_y1 + BUTTON_H

row2_y1 = row1_y2 + 6
row2_y2 = row2_y1 + BUTTON_H

TOOLBAR_HEIGHT = row2_y2 + 8

save_count = 1
shape_mode = False  # Toggle for AI shape auto-correction

# =============================================================================
# AI Features
# =============================================================================
smoother_index = SmoothLandmark(alpha=0.4)
stroke_smoother = StrokeSmoother()
shape_recognizer = ShapeRecognizer()
stats = DrawingStats()

# Minimum movement threshold to filter jitter (in pixels)
MIN_MOVE_THRESHOLD = 3

# Current stroke points (for shape recognition)
current_stroke = []
is_drawing = False

# Gesture cooldowns
undo_cooldown = 0
redo_cooldown = 0
GESTURE_COOLDOWN = 1.0  # seconds

# Frame timestamp counter for VIDEO mode
frame_timestamp_ms = 0

# =============================================================================
# Draw Toolbar
# =============================================================================
def draw_toolbar(img):
    """Draw the toolbar buttons on the image."""
    # Dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (screen_w, TOOLBAR_HEIGHT), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # Bottom border glow
    cv2.line(img, (0, TOOLBAR_HEIGHT), (screen_w, TOOLBAR_HEIGHT), (108, 92, 231), 2)

    for r, row in enumerate(TOOLS):
        sx = start_x_row[r]
        for i, tool in enumerate(row):

            x1 = sx + i * (BUTTON_W + GAP)
            x2 = x1 + BUTTON_W

            if r == 0:
                y1, y2 = row1_y1, row1_y2
            else:
                y1, y2 = row2_y1, row2_y2

            # Determine button color
            if tool in COLORS:
                color = COLORS[tool]
            elif tool == "ERASE":
                color = (60, 60, 60)
            elif tool == "CLEAR":
                color = (100, 100, 100)
            elif tool == "UNDO":
                color = (180, 120, 0)
            elif tool == "REDO":
                color = (0, 120, 180)
            elif tool == "SHAPES":
                color = (200, 50, 200) if shape_mode else (80, 40, 80)
            elif tool == "SAVE":
                color = (0, 140, 255)
            elif tool == "EXIT":
                color = (40, 40, 160)
            else:
                color = (80, 80, 80)

            # Draw button
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

            # Highlight active color
            if tool in COLORS and COLORS[tool] == draw_color and draw_color != BLACK:
                cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)

            # Button label
            text_size = cv2.getTextSize(tool, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            text_x = x1 + (BUTTON_W - text_size[0]) // 2
            text_y = y1 + (BUTTON_H + text_size[1]) // 2
            cv2.putText(img, tool, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

# =============================================================================
# Draw Gesture Indicator
# =============================================================================
def draw_gesture_indicator(img, gesture):
    """Show current gesture mode in the bottom-left corner."""
    gesture_colors = {
        'DRAW': (50, 220, 50),
        'SELECT': (255, 200, 0),
        'UNDO': (0, 140, 255),
        'REDO': (255, 100, 0),
        'IDLE': (100, 100, 100),
    }
    color = gesture_colors.get(gesture, (100, 100, 100))
    label = f"Mode: {gesture}"

    # Background
    cv2.rectangle(img, (10, screen_h - 40), (200, screen_h - 10), (20, 20, 20), -1)
    cv2.rectangle(img, (10, screen_h - 40), (200, screen_h - 10), color, 1)
    cv2.putText(img, label, (20, screen_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# =============================================================================
# Draw Hand Landmarks manually (new API gives NormalizedLandmark lists)
# =============================================================================
def draw_hand_on_frame(img, hand_landmarks, w, h):
    """Draw hand landmarks and connections on the frame."""
    points = []
    for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(img, (px, py), 3, (0, 255, 200), -1)

    # Draw connections
    connections = HandLandmarksConnections.HAND_CONNECTIONS
    for conn in connections:
        start_idx = conn.start
        end_idx = conn.end
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(img, points[start_idx], points[end_idx], (0, 200, 180), 1)

# =============================================================================
# Main Loop
# =============================================================================
while True:

    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    draw_toolbar(img)

    h, w, _ = img.shape

    # Convert to MediaPipe Image for the new Tasks API
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Detect hand landmarks (VIDEO mode requires timestamp)
    frame_timestamp_ms += 33  # ~30 fps
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    current_gesture = 'IDLE'

    if result.hand_landmarks:

        for hand_idx, hand_lms in enumerate(result.hand_landmarks):

            # Build landmark list as (x_pixel, y_pixel)
            lm_list = []
            for lm in hand_lms:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Draw hand on frame
            draw_hand_on_frame(img, hand_lms, w, h)

            # Get finger states and gesture
            fingers = count_fingers_up(lm_list)
            gesture = get_gesture(fingers)
            current_gesture = gesture

            # Index finger tip (smoothed)
            raw_tip = lm_list[8]
            x1, y1 = smoother_index.update(raw_tip)

            # -----------------------------------------------------------------
            # TOOL SELECTION (SELECT gesture in toolbar area)
            # -----------------------------------------------------------------
            if gesture in ('DRAW', 'SELECT') and y1 < TOOLBAR_HEIGHT:
                xp, yp = 0, 0

                for r, row in enumerate(TOOLS):
                    sx = start_x_row[r]
                    for i, tool in enumerate(row):
                        bx1 = sx + i * (BUTTON_W + GAP)
                        bx2 = bx1 + BUTTON_W

                        if r == 0:
                            by1, by2 = row1_y1, row1_y2
                        else:
                            by1, by2 = row2_y1, row2_y2

                        if bx1 < x1 < bx2 and by1 < y1 < by2:

                            if tool in COLORS:
                                draw_color = COLORS[tool]

                            elif tool == "ERASE":
                                draw_color = BLACK

                            elif tool == "CLEAR":
                                save_state()
                                canvas = np.zeros((screen_h, screen_w, 3), np.uint8)

                            elif tool == "UNDO":
                                undo()
                                time.sleep(0.3)

                            elif tool == "REDO":
                                redo()
                                time.sleep(0.3)

                            elif tool == "SHAPES":
                                shape_mode = not shape_mode
                                time.sleep(0.3)

                            elif tool == "SAVE":
                                filename = f"drawing_{save_count}.png"
                                cv2.imwrite(filename, canvas)
                                print("Saved:", filename)
                                save_count += 1
                                time.sleep(0.4)

                            elif tool == "EXIT":
                                cap.release()
                                cv2.destroyAllWindows()
                                exit()

            # -----------------------------------------------------------------
            # DRAWING MODE (index finger only)
            # -----------------------------------------------------------------
            elif gesture == 'DRAW' and y1 >= TOOLBAR_HEIGHT:

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                    if not is_drawing:
                        save_state()
                        is_drawing = True
                        current_stroke = [(x1, y1)]

                # Jitter filter: only draw if moved enough
                dist = np.sqrt((x1 - xp) ** 2 + (y1 - yp) ** 2)
                if dist >= MIN_MOVE_THRESHOLD:
                    thickness = eraser_thickness if draw_color == BLACK else brush_thickness
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
                    xp, yp = x1, y1
                    current_stroke.append((x1, y1))

                    if draw_color == BLACK:
                        stats.add_stroke(draw_color)

            # -----------------------------------------------------------------
            # GESTURE: UNDO (Fist)
            # -----------------------------------------------------------------
            elif gesture == 'UNDO':
                if time.time() - undo_cooldown > GESTURE_COOLDOWN:
                    undo()
                    undo_cooldown = time.time()
                xp, yp = 0, 0

            # -----------------------------------------------------------------
            # GESTURE: REDO (Open Palm)
            # -----------------------------------------------------------------
            elif gesture == 'REDO':
                if time.time() - redo_cooldown > GESTURE_COOLDOWN:
                    redo()
                    redo_cooldown = time.time()
                xp, yp = 0, 0

            # -----------------------------------------------------------------
            # IDLE — stop drawing
            # -----------------------------------------------------------------
            else:
                if is_drawing and len(current_stroke) > 5 and draw_color != BLACK:
                    stats.add_stroke(draw_color)

                    # Shape auto-correction
                    if shape_mode and len(current_stroke) > 10:
                        pts = np.array(current_stroke, dtype=np.int32).reshape((-1, 1, 2))
                        result_shape = shape_recognizer.recognize(pts)
                        if result_shape['type'] != 'unknown':
                            # Clear the raw stroke and draw perfect shape
                            canvas = undo_stack[-1].copy() if undo_stack else np.zeros((screen_h, screen_w, 3), np.uint8)
                            canvas, shape_type = auto_correct_shape(canvas, pts, draw_color, brush_thickness)
                            stats.add_shape(shape_type)

                is_drawing = False
                current_stroke = []
                xp, yp = 0, 0

    else:
        # No hand detected
        if is_drawing and len(current_stroke) > 5 and draw_color != BLACK:
            stats.add_stroke(draw_color)
        is_drawing = False
        current_stroke = []
        xp, yp = 0, 0

    # =========================================================================
    # Merge Canvas onto Camera Feed
    # =========================================================================
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # Draw overlays
    draw_gesture_indicator(img, current_gesture)
    stats.draw_overlay(img)

    # Shape mode indicator
    if shape_mode:
        cv2.putText(img, "SHAPE AI: ON", (screen_w - 200, screen_h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 200), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()