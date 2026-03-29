"""
ISRO IROUC 2026 — QR Drone Detector with Seed Matching & Orientation Feedback
Single file version - Copy and run directly
"""

import cv2
import numpy as np
import threading
import time
import hashlib
import math
from collections import deque
from tkinter import filedialog, Tk


# ═══════════════════════════════════════════════════════════════
#  SEED IMAGE MATCHER
# ═══════════════════════════════════════════════════════════════
class SeedMatcher:
    def __init__(self):
        self.seed_image = None
        self.seed_hash = None
        self.seed_data = None
        self.seed_loaded = False
        self.seed_features = None
        self.matching_threshold = 85
        
    def load_seed_from_file(self, filepath):
        try:
            img = cv2.imread(filepath)
            if img is None:
                return False, "Could not read image file"
            
            self.seed_image = img.copy()
            detector = cv2.QRCodeDetector()
            data, pts, _ = detector.detectAndDecode(img)
            
            if data and len(data) > 0:
                self.seed_data = data
                self.seed_hash = hashlib.sha256(data.encode()).hexdigest()
                self.seed_loaded = True
                self._extract_seed_features(img, pts)
                return True, f"Loaded: {data[:50]}..."
            else:
                return False, "No QR code found in seed image"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _extract_seed_features(self, img, qr_pts):
        if qr_pts is None or len(qr_pts) == 0:
            self.seed_features = None
            return
        
        corners = qr_pts[0].astype(np.float32)
        top_edge = corners[1] - corners[0]
        angle = math.degrees(math.atan2(top_edge[1], top_edge[0]))
        
        self.seed_features = {
            'angle': angle,
            'corners': corners,
            'aspect_ratio': self._calculate_aspect_ratio(corners)
        }
    
    def _calculate_aspect_ratio(self, corners):
        top_edge = np.linalg.norm(corners[1] - corners[0])
        left_edge = np.linalg.norm(corners[3] - corners[0])
        return top_edge / left_edge if left_edge > 0 else 1.0
    
    def compare_with_seed(self, qr_data):
        if not self.seed_loaded or not qr_data:
            return False, 0
        
        if qr_data == self.seed_data:
            return True, 100
        
        similarity = self._calculate_similarity(qr_data, self.seed_data)
        return similarity >= self.matching_threshold, similarity
    
    def _calculate_similarity(self, str1, str2):
        if not str1 or not str2:
            return 0
        
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 100
        
        matches = sum(1 for i in range(min(len(str1), len(str2))) if str1[i] == str2[i])
        return (matches / max_len) * 100
    
    def analyze_orientation(self, qr_corners):
        if not self.seed_loaded or qr_corners is None or self.seed_features is None:
            return None
        
        try:
            corners = qr_corners.astype(np.float32)
            top_edge = corners[1] - corners[0]
            current_angle = math.degrees(math.atan2(top_edge[1], top_edge[0]))
            current_aspect = self._calculate_aspect_ratio(corners)
            
            angle_diff = (current_angle - self.seed_features['angle'] + 180) % 360 - 180
            aspect_diff = current_aspect - self.seed_features['aspect_ratio']
            
            is_angle_ok = abs(angle_diff) <= 15
            is_perspective_ok = abs(aspect_diff) <= 0.2
            
            feedback = []
            if not is_angle_ok:
                direction = "clockwise" if angle_diff > 0 else "counter-clockwise"
                feedback.append(f"Rotate {abs(angle_diff):.0f}° {direction}")
            
            if not is_perspective_ok:
                feedback.append("Tilt camera " + ("upward" if aspect_diff > 0 else "downward"))
            
            skew = self._calculate_skew(corners)
            if skew > 0.15:
                feedback.append("Hold QR flat (reduce skew)")
            
            return {
                'current_angle': current_angle,
                'target_angle': self.seed_features['angle'],
                'angle_diff': angle_diff,
                'is_angle_ok': is_angle_ok,
                'is_perspective_ok': is_perspective_ok,
                'feedback': feedback,
                'skew': skew
            }
        except Exception:
            return None
    
    def _calculate_skew(self, corners):
        left_side = corners[3] - corners[0]
        right_side = corners[2] - corners[1]
        
        angle_left = math.atan2(left_side[1], left_side[0])
        angle_right = math.atan2(right_side[1], right_side[0])
        skew = abs(math.degrees(angle_left - angle_right)) / 90.0
        
        return min(skew, 1.0)
    
    def reset(self):
        self.seed_image = None
        self.seed_hash = None
        self.seed_data = None
        self.seed_loaded = False
        self.seed_features = None


# ═══════════════════════════════════════════════════════════════
#  THREADED CAMERA
# ═══════════════════════════════════════════════════════════════
class CameraStream:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        self.ret, self.frame = self.cap.read()
        self._lock = threading.Lock()
        self.running = True
        self.cam_fps = 0.0
        self._ftimes = deque(maxlen=30)
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            t = time.time()
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame
                self._ftimes.append(t)
                if len(self._ftimes) >= 2:
                    self.cam_fps = (len(self._ftimes)-1) / (
                        self._ftimes[-1] - self._ftimes[0] + 1e-6)

    def read(self):
        with self._lock:
            if not self.ret or self.frame is None:
                return False, None, 0.0
            return True, self.frame.copy(), self.cam_fps

    def stop(self):
        self.running = False
        self.cap.release()


# ═══════════════════════════════════════════════════════════════
#  QR WORKER
# ═══════════════════════════════════════════════════════════════
class QRWorker:
    def __init__(self, seed_matcher=None):
        self.detector = cv2.QRCodeDetector()
        self.seed_matcher = seed_matcher
        self._lock = threading.Lock()
        self._frame = None
        self._result = (None, None, None, None)
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _try(self, img):
        try:
            data, pts, _ = self.detector.detectAndDecode(img)
            if pts is not None and len(pts) > 0 and pts[0].shape[0] == 4:
                return data or "", pts[0]
        except:
            pass
        return None, None

    def _run(self):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        while self.running:
            with self._lock:
                frame = self._frame
            if frame is None:
                time.sleep(0.001)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = (None, None)
            match_info = None
            orientation_info = None

            # Strategy 1: Plain gray
            data, pts = self._try(gray)
            if pts is not None:
                result = (data, pts)

            # Strategy 2: CLAHE
            if result[1] is None:
                data, pts = self._try(clahe.apply(gray))
                if pts is not None:
                    result = (data, pts)

            # Strategy 3: Sharpen
            if result[1] is None:
                data, pts = self._try(cv2.filter2D(gray, -1, k_sharp))
                if pts is not None:
                    result = (data, pts)

            # Strategy 4: Otsu
            if result[1] is None:
                _, th = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                data, pts = self._try(th)
                if pts is not None:
                    result = (data, pts)

            # Strategy 5: Downscale
            if result[1] is None:
                s = cv2.resize(gray, None, fx=0.6, fy=0.6)
                d, p = self._try(s)
                if p is not None:
                    result = (d, p / 0.6)

            # Strategy 6: Upscale
            if result[1] is None:
                s = cv2.resize(gray, None, fx=1.8, fy=1.8)
                d, p = self._try(s)
                if p is not None:
                    result = (d, p / 1.8)

            # Check match with seed
            if result[1] is not None and self.seed_matcher and result[0]:
                is_match, similarity = self.seed_matcher.compare_with_seed(result[0])
                if is_match:
                    match_info = {
                        'is_match': True,
                        'similarity': similarity,
                        'seed_data': self.seed_matcher.seed_data
                    }
                    orientation_info = self.seed_matcher.analyze_orientation(result[1])
                else:
                    match_info = {
                        'is_match': False,
                        'similarity': similarity,
                        'seed_data': None
                    }

            with self._lock:
                self._result = (result[0], result[1], match_info, orientation_info)
                self._frame = None

    def submit(self, frame):
        with self._lock:
            if self._frame is None:
                self._frame = frame

    def get(self):
        with self._lock:
            return self._result

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
#  TRACKER
# ═══════════════════════════════════════════════════════════════
class SimpleTracker:
    def __init__(self, smooth_factor=0.65):
        self.smooth_factor = smooth_factor
        self.last_pts = None
        self.has_track = False
    
    def update(self, pts):
        if pts is None:
            return None
        
        pts = pts.astype(np.float32)
        
        if not self.has_track:
            self.last_pts = pts
            self.has_track = True
            return pts
        
        smoothed = self.smooth_factor * pts + (1 - self.smooth_factor) * self.last_pts
        self.last_pts = smoothed
        return smoothed
    
    def reset(self):
        self.last_pts = None
        self.has_track = False


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def order_points(pts):
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype=np.float32)


def is_valid_qr(pts, fw, fh):
    pts[:, 0] = np.clip(pts[:, 0], 0, fw - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, fh - 1)
    sides = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]
    mn, mx = min(sides), max(sides)
    if mn < 25 or mx > 900:
        return False
    if mx / (mn + 1e-5) > 4.5:
        return False
    area = cv2.contourArea(pts.astype(np.float32))
    if area < 500 or area > 600000:
        return False
    return True


def draw_overlay(frame, pts, last_data, match_info=None, orientation_info=None):
    p = pts.astype(np.int32)
    fh, fw = frame.shape[:2]

    if match_info and match_info.get('is_match', False):
        if orientation_info and not orientation_info.get('is_angle_ok', True):
            border_color = (0, 165, 255)
            fill_color = (0, 100, 150)
            marker_color = (0, 165, 255)
        else:
            border_color = (0, 255, 0)
            fill_color = (0, 150, 0)
            marker_color = (0, 255, 0)
    elif last_data:
        border_color = (0, 0, 255)
        fill_color = (0, 0, 150)
        marker_color = (0, 0, 255)
    else:
        border_color = (100, 100, 100)
        fill_color = (100, 100, 100)
        marker_color = (100, 100, 100)

    # Tint
    ov = frame.copy()
    cv2.fillPoly(ov, [p], fill_color)
    cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

    # Edges
    for i in range(4):
        cv2.line(frame, tuple(p[i]), tuple(p[(i+1)%4]),
                 border_color, 2, cv2.LINE_AA)

    # Corners
    dirs = [(1,1), (-1,1), (-1,-1), (1,-1)]
    for i, (sx, sy) in enumerate(dirs):
        x, y = int(p[i][0]), int(p[i][1])
        L = 20
        cv2.line(frame, (x, y), (x + sx * L, y), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + sy * L), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 6, marker_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 9, (255, 255, 255), 1, cv2.LINE_AA)

    # Centre
    cx = int(np.mean(p[:, 0]))
    cy = int(np.mean(p[:, 1]))
    cv2.drawMarker(frame, (cx, cy), marker_color, cv2.MARKER_CROSS, 34, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 17, marker_color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 4, marker_color, -1, cv2.LINE_AA)

    # Label
    if last_data:
        if match_info and match_info.get('is_match', False):
            if orientation_info and not orientation_info.get('is_angle_ok', True):
                label = f"⚠ ADJUST ORIENTATION"
                label_color = (0, 165, 255)
            else:
                label = f"✓ MATCHED: {last_data[:25]}"
                label_color = (0, 255, 0)
        else:
            label = f"QR: {last_data[:25]}"
            label_color = (0, 0, 255)
        
        tx = max(min(p[0][0], fw - 250), 6)
        ty = max(p[0][1] - 14, 22)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (tx - 6, ty - th - 8), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
        cv2.rectangle(frame, (tx - 6, ty - th - 8), (tx + tw + 6, ty + 6), label_color, 1)
        cv2.putText(frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)


def load_seed_image_dialog():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Select Seed QR Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path if file_path else None


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("═"*60)
    print("  ISRO IROUC 2026 — QR Drone Detector with Orientation Feedback")
    print("  Q to quit | L to load seed image | R to reset seed")
    print("═"*60 + "\n")
    
    seed_matcher = SeedMatcher()
    
    print("Please select a seed QR image...")
    seed_path = load_seed_image_dialog()
    
    if seed_path:
        success, message = seed_matcher.load_seed_from_file(seed_path)
        if success:
            print(f"✓ {message}")
            if seed_matcher.seed_features:
                print(f"  Reference orientation: {seed_matcher.seed_features['angle']:.1f}°")
        else:
            print(f"✗ {message}")
    else:
        print("No seed loaded. Running without matching...")
    
    print("\nStarting camera...")
    
    cam = CameraStream(0)
    worker = QRWorker(seed_matcher)
    tracker = SimpleTracker()
    
    last_data = ""
    last_match_info = None
    last_orientation = None
    lost_frames = 0
    display_pts = None
    LOSE_AFTER = 30
    
    last_feedback_time = 0
    feedback_cooldown = 1.0
    
    while True:
        ret, frame, cam_fps = cam.read()
        if not ret or frame is None:
            continue
        
        fh, fw = frame.shape[:2]
        
        worker.submit(frame)
        data, raw_pts, match_info, orientation_info = worker.get()
        
        if match_info and match_info.get('is_match', False):
            last_match_info = match_info
            last_orientation = orientation_info
            
            current_time = time.time()
            if orientation_info and (current_time - last_feedback_time) > feedback_cooldown:
                if not orientation_info.get('is_angle_ok', True):
                    print("\n" + "="*50)
                    print("⚠ ORIENTATION FEEDBACK ⚠")
                    print("="*50)
                    for fb in orientation_info.get('feedback', []):
                        print(f"  → {fb}")
                    print(f"  Current angle: {orientation_info['current_angle']:.1f}°")
                    print(f"  Target angle: {orientation_info['target_angle']:.1f}°")
                    print("="*50 + "\n")
                    last_feedback_time = current_time
        
        if raw_pts is not None:
            ordered = order_points(raw_pts.astype(np.float32))
            if is_valid_qr(ordered, fw, fh):
                display_pts = tracker.update(ordered)
                lost_frames = 0
                if data and data != last_data:
                    if match_info and match_info.get('is_match', False):
                        print(f"  [✓ MATCHED ✓]  {data}")
                    else:
                        print(f"  [QR]  {data}")
                    last_data = data
        else:
            lost_frames += 1
            if lost_frames >= LOSE_AFTER:
                display_pts = None
                last_data = ""
                last_match_info = None
                last_orientation = None
                tracker.reset()
        
        # Draw
        if display_pts is not None:
            draw_overlay(frame, display_pts, last_data, last_match_info, last_orientation)
            
            if last_match_info and last_match_info.get('is_match', False):
                if last_orientation and not last_orientation.get('is_angle_ok', True):
                    status = "⚠ ADJUST ORIENTATION ⚠"
                    color = (0, 165, 255)
                else:
                    status = "✓ SEED MATCHED ✓"
                    color = (0, 255, 0)
            else:
                status = "QR LOCKED"
                color = (0, 0, 255)
        else:
            status = "Scanning..."
            color = (160, 160, 160)
        
        # UI
        cv2.rectangle(frame, (0, 0), (fw, 70), (0, 0, 0), -1)
        cv2.putText(frame, status, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
        
        if seed_matcher.seed_loaded:
            seed_short = seed_matcher.seed_data[:30] + "..."
            cv2.putText(frame, f"SEED: {seed_short}", (12, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"CAM: {cam_fps:.0f} fps", (fw - 210, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, "Q:Quit L:Load R:Reset", (fw - 350, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "ISRO IROUC 2026 — QR Detector",
                    (12, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.imshow("ISRO IROUC — QR Drone Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            seed_path = load_seed_image_dialog()
            if seed_path:
                success, message = seed_matcher.load_seed_from_file(seed_path)
                if success:
                    print(f"✓ {message}")
                    lost_frames = LOSE_AFTER + 1
                    last_match_info = None
                    last_orientation = None
                else:
                    print(f"✗ {message}")
        elif key == ord('r'):
            seed_matcher.reset()
            last_match_info = None
            last_orientation = None
            print("Seed matcher reset")
    
    worker.stop()
    cam.stop()
    cv2.destroyAllWindows()
    print("\nDetector closed.")


if __name__ == "__main__":
    main()