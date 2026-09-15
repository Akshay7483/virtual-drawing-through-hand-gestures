import cv2
import numpy as np
import math
import time
from scipy.ndimage import gaussian_filter1d

class ShapeRecognizer:
    """Classifies drawn shapes from contours."""
    def __init__(self):
        pass

    def recognize(self, contour):
        """
        Recognizes the shape of a given contour.
        Returns a dict: {'type': 'circle'|'rectangle'|'triangle'|'line'|'unknown', 'params': {...}}
        """
        perimeter = cv2.arcLength(contour, True)
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        num_vertices = len(approx)

        if num_vertices == 3:
            return {
                'type': 'triangle',
                'params': {'points': [tuple(pt[0]) for pt in approx]}
            }
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            return {
                'type': 'rectangle',
                'params': {'x': x, 'y': y, 'w': w, 'h': h}
            }
        elif num_vertices > 6:
            area = cv2.contourArea(contour)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.7:  # Threshold for circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    return {
                        'type': 'circle',
                        'params': {'center': (int(x), int(y)), 'radius': int(radius)}
                    }
        
        # Check for line (very elongated or 2 endpoints)
        if num_vertices >= 2:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:
                aspect_ratio = float(w) / h
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    contour_squeeze = np.squeeze(contour)
                    if len(contour_squeeze.shape) == 2 and len(contour_squeeze) >= 2:
                        # Find the two furthest points
                        dist_matrix = np.linalg.norm(contour_squeeze[:, None] - contour_squeeze, axis=-1)
                        i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
                        pt1 = tuple(int(x) for x in contour_squeeze[i])
                        pt2 = tuple(int(x) for x in contour_squeeze[j])
                        return {
                            'type': 'line',
                            'params': {'start': pt1, 'end': pt2}
                        }

        return {'type': 'unknown', 'params': {}}

def auto_correct_shape(canvas, contour, color, thickness):
    """
    Uses ShapeRecognizer to identify shape and draw a perfect version on canvas.
    Returns (canvas, shape_type_string).
    """
    recognizer = ShapeRecognizer()
    result = recognizer.recognize(contour)
    shape_type = result['type']
    params = result['params']

    if shape_type == 'circle':
        cv2.circle(canvas, params['center'], params['radius'], color, thickness)
    elif shape_type == 'rectangle':
        x, y, w, h = params['x'], params['y'], params['w'], params['h']
        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, thickness)
    elif shape_type == 'triangle':
        pts = np.array(params['points'], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
    elif shape_type == 'line':
        cv2.line(canvas, params['start'], params['end'], color, thickness)
        
    return canvas, shape_type

class StrokeSmoother:
    """Smooths out drawn strokes using Gaussian filtering."""
    def __init__(self):
        pass

    def smooth(self, points, sigma=3):
        """
        Smooths a list of (x,y) points using scipy.ndimage.gaussian_filter1d.
        Returns smoothed points.
        """
        if len(points) < 3:
            return points

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        smoothed_x = gaussian_filter1d(x_coords, sigma=sigma)
        smoothed_y = gaussian_filter1d(y_coords, sigma=sigma)

        return [(int(x), int(y)) for x, y in zip(smoothed_x, smoothed_y)]

class DrawingStats:
    """Tracks statistics for the drawing session."""
    def __init__(self):
        self.stroke_count = 0
        self.colors_used = set()
        self.start_time = time.time()
        self.shapes_recognized = {}

    def add_stroke(self, color):
        """Records a new stroke and its color."""
        self.stroke_count += 1
        self.colors_used.add(tuple(color))

    def add_shape(self, shape_type):
        """Records a recognized shape."""
        if shape_type != 'unknown':
            self.shapes_recognized[shape_type] = self.shapes_recognized.get(shape_type, 0) + 1

    def get_elapsed_time(self):
        """Returns elapsed time as 'MM:SS' string."""
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        return f"{mins:02d}:{secs:02d}"

    def get_summary(self):
        """Returns a dict summary of stats."""
        return {
            'stroke_count': self.stroke_count,
            'colors_used': len(self.colors_used),
            'elapsed_time': self.get_elapsed_time(),
            'shapes_recognized': self.shapes_recognized
        }

    def draw_overlay(self, img, x=10, y_start=None):
        """Draws statistics overlay on the image."""
        if y_start is None:
            h = img.shape[0]
            y_start = h - 120
            
        stats = [
            f"Time: {self.get_elapsed_time()}",
            f"Strokes: {self.stroke_count}",
            f"Colors: {len(self.colors_used)}"
        ]
        
        shapes_str = ", ".join([f"{k}:{v}" for k, v in self.shapes_recognized.items()])
        if shapes_str:
            stats.append(f"Shapes: {shapes_str}")

        y_offset = y_start
        for stat in stats:
            cv2.putText(img, stat, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
