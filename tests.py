"""
ISRO IROUC 2026 — QR Drone Detector with Webcam Support
Long-range QR detection + Terminal rotation instructions + Drone control
"""

import cv2
import numpy as np
import threading
import time
import math
import socket
from collections import deque
from tkinter import filedialog, Tk
import hashlib
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
#  DRONE CONTROLLER
# ═══════════════════════════════════════════════════════════════
class DroneController:
    def __init__(self, drone_ip="192.168.1.100", port=8888):
        self.drone_ip = drone_ip
        self.port = port
        self.connected = False
        self.sock = None
        self.control_enabled = False
        self.search_mode = True
        self.last_command_time = 0
        self.command_cooldown = 0.3
        
    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1)
            self.connected = True
            print(f"✓ Drone connected to {self.drone_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Drone connection failed: {e}")
            return False
    
    def send_command(self, command):
        if not self.connected or not self.control_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_command_time >= self.command_cooldown:
            self.last_command_time = current_time
            try:
                self.sock.sendto(command.encode(), (self.drone_ip, self.port))
                print(f"  🚁 Drone: {command}")
            except Exception as e:
                print(f"Command failed: {e}")
    
    def search_for_qr(self):
        if not self.search_mode:
            return
        self.send_command("ROTATE_RIGHT 20")
    
    def navigate_to_qr(self, qr_center, frame_center, orientation_info=None):
        if not self.control_enabled:
            return
        
        error_x = qr_center[0] - frame_center[0]
        error_y = qr_center[1] - frame_center[1]
        
        # PRIORITY 1: FIX QR ROTATION
        if orientation_info and not orientation_info.get('is_angle_ok', True):
            angle_diff = orientation_info.get('angle_diff', 0)
            rotation_amount = min(max(abs(angle_diff) // 5, 5), 30)
            
            if abs(angle_diff) > 5:
                if angle_diff > 0:
                    self.send_command(f"ROTATE_RIGHT {rotation_amount}")
                else:
                    self.send_command(f"ROTATE_LEFT {rotation_amount}")
                return
        
        # PRIORITY 2: CENTER
        if abs(error_x) > 60:
            move = min(abs(error_x) // 20, 25)
            if error_x > 0:
                self.send_command(f"MOVE_RIGHT {move}")
            else:
                self.send_command(f"MOVE_LEFT {move}")
            return
        
        if abs(error_y) > 60:
            move = min(abs(error_y) // 20, 25)
            if error_y > 0:
                self.send_command(f"MOVE_DOWN {move}")
            else:
                self.send_command(f"MOVE_UP {move}")
            return
        
        if self.search_mode:
            self.search_mode = False
            print("  ✓ QR aligned! Hovering...")
        self.send_command("HOVER")
    
    def land(self):
        self.send_command("LAND")
    
    def disconnect(self):
        self.control_enabled = False
        if self.sock:
            self.sock.close()
        self.connected = False


# ═══════════════════════════════════════════════════════════════
#  SEED IMAGE MATCHER
# ═══════════════════════════════════════════════════════════════
class SeedMatcher:
    def __init__(self):
        self.seed_image = None
        self.seed_data = None
        self.seed_loaded = False
        self.seed_features = None
        self.matching_threshold = 85
        self.calibration_samples = []
        self.is_calibrated = False
        self.calibrated_angle = None
        self.last_angle_message_time = 0
        self.message_cooldown = 1.5
        
    def load_seed_from_file(self, filepath):
        try:
            img = cv2.imread(filepath)
            if img is None:
                return False, "Could not read image file"
            
            img = cv2.resize(img, (400, 400))
            
            self.seed_image = img.copy()
            detector = cv2.QRCodeDetector()
            data, pts, _ = detector.detectAndDecode(img)
            
            if data and len(data) > 0:
                self.seed_data = data
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
    
    def auto_calibrate(self, qr_corners):
        if qr_corners is None:
            return False
        
        corners = qr_corners.astype(np.float32)
        top_edge = corners[1] - corners[0]
        current_angle = math.degrees(math.atan2(top_edge[1], top_edge[0]))
        
        self.calibration_samples.append(current_angle)
        
        if len(self.calibration_samples) > 10:
            self.calibration_samples.pop(0)
        
        if len(self.calibration_samples) >= 5 and not self.is_calibrated:
            self.calibrated_angle = np.median(self.calibration_samples)
            if self.seed_features:
                self.seed_features['angle'] = self.calibrated_angle
            self.is_calibrated = True
            print(f"\n✅ AUTO-CALIBRATION COMPLETE!")
            print(f"   Reference angle: {self.calibrated_angle:.1f}°\n")
            return True
        
        return False
    
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
            
            angle_diff = (current_angle - self.seed_features['angle'] + 180) % 360 - 180
            is_angle_ok = abs(angle_diff) <= 10
            
            rotation_direction = None
            rotation_amount = abs(angle_diff)
            
            if not is_angle_ok:
                rotation_direction = "clockwise" if angle_diff > 0 else "counter-clockwise"
                
                current_time = time.time()
                if current_time - self.last_angle_message_time >= self.message_cooldown:
                    self.last_angle_message_time = current_time
                    print(f"\n{'='*50}")
                    print(f"⚠️  ROTATION INSTRUCTION ⚠️")
                    print(f"{'='*50}")
                    print(f"  Current QR angle: {current_angle:.1f}°")
                    print(f"  Target angle:     {self.seed_features['angle']:.1f}°")
                    print(f"  Difference:       {abs(angle_diff):.1f}°")
                    if angle_diff > 0:
                        print(f"  ▶  Rotate QR CLOCKWISE by {abs(angle_diff):.0f}°")
                        print(f"  ▶  Drone command: ROTATE_RIGHT {min(max(abs(angle_diff)//5, 5), 30)}°")
                    else:
                        print(f"  ◀  Rotate QR COUNTER-CLOCKWISE by {abs(angle_diff):.0f}°")
                        print(f"  ◀  Drone command: ROTATE_LEFT {min(max(abs(angle_diff)//5, 5), 30)}°")
                    print(f"{'='*50}\n")
            
            return {
                'current_angle': current_angle,
                'target_angle': self.seed_features['angle'],
                'angle_diff': angle_diff,
                'rotation_direction': rotation_direction,
                'rotation_amount': rotation_amount,
                'is_angle_ok': is_angle_ok
            }
        except Exception:
            return None
    
    def reset(self):
        self.seed_image = None
        self.seed_data = None
        self.seed_loaded = False
        self.seed_features = None
        self.calibration_samples = []
        self.is_calibrated = False


# ═══════════════════════════════════════════════════════════════
#  ENHANCED QR DETECTOR WITH WEBCAM OPTIMIZATION
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

    def _try_detect(self, img, scale=1.0):
        try:
            if scale != 1.0:
                h, w = img.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                img_scaled = cv2.resize(img, (new_w, new_h))
                data, pts, _ = self.detector.detectAndDecode(img_scaled)
                if pts is not None and len(pts) > 0 and pts[0].shape[0] == 4:
                    pts = pts / scale
                    return data or "", pts[0]
            else:
                data, pts, _ = self.detector.detectAndDecode(img)
                if pts is not None and len(pts) > 0 and pts[0].shape[0] == 4:
                    return data or "", pts[0]
        except:
            pass
        return None, None

    def _run(self):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        scales = [2.0, 1.5, 1.2, 1.0, 0.8, 0.6]
        
        while self.running:
            with self._lock:
                frame = self._frame
            if frame is None:
                time.sleep(0.001)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = clahe.apply(gray)
            sharpened = cv2.filter2D(gray, -1, k_sharp)
            
            result = (None, None)
            match_info = None
            orientation_info = None
            
            for scale in scales:
                if result[1] is not None:
                    break
                
                data, pts = self._try_detect(gray, scale)
                if pts is not None:
                    result = (data, pts)
                    break
                
                data, pts = self._try_detect(enhanced, scale)
                if pts is not None:
                    result = (data, pts)
                    break
                
                data, pts = self._try_detect(sharpened, scale)
                if pts is not None:
                    result = (data, pts)
                    break

            if result[1] is None:
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                data, pts = self._try_detect(th, 1.0)
                if pts is not None:
                    result = (data, pts)

            if result[1] is not None and self.seed_matcher and result[0]:
                is_match, similarity = self.seed_matcher.compare_with_seed(result[0])
                if is_match:
                    match_info = {'is_match': True, 'similarity': similarity}
                    orientation_info = self.seed_matcher.analyze_orientation(result[1])
                else:
                    match_info = {'is_match': False, 'similarity': similarity}

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
#  WEBCAM STREAM WITH CAMERA SELECTION
# ═══════════════════════════════════════════════════════════════
class WebcamStream:
    def __init__(self, camera_index=1):  # CHANGED: Default to index 1 (external webcam)
        self.camera_index = camera_index
        print(f"\n📷 Trying to open camera index {camera_index}...")
        
        # Try different backends for better compatibility
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"   Camera {camera_index} not found with DirectShow, trying default...")
            self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {camera_index}")
            print(f"   Available cameras:")
            # Try to find available cameras
            for i in range(5):
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    print(f"   - Camera index {i} is available")
                    test_cap.release()
            raise Exception(f"No camera found at index {camera_index}")
        
        # Set optimal webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Try to set MJPG format for better performance
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        self.ret, self.frame = self.cap.read()
        self._lock = threading.Lock()
        self.running = True
        self.cam_fps = 0.0
        self._ftimes = deque(maxlen=30)
        threading.Thread(target=self._update, daemon=True).start()
        print(f"✅ Webcam {camera_index} initialized successfully!")

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
#  TRACKER
# ═══════════════════════════════════════════════════════════════
class SimpleTracker:
    def __init__(self, smooth_factor=0.7):
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
    
    if mn < 15 or mx > 1000:
        return False
    if mx / (mn + 1e-5) > 5.0:
        return False
    
    return True


def draw_rotation_arrow(frame, center, direction, angle_diff):
    cx, cy = center
    radius = 70
    arrow_color = (0, 165, 255)
    
    if direction == "clockwise":
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, 270, arrow_color, 3)
        end_angle = math.radians(270)
        tip_x = int(cx + radius * math.cos(end_angle))
        tip_y = int(cy + radius * math.sin(end_angle))
        cv2.arrowedLine(frame, (tip_x, tip_y), (tip_x - 20, tip_y - 20), arrow_color, 3, cv2.LINE_AA)
        cv2.putText(frame, f"ROTATE RIGHT {abs(angle_diff):.0f}°", (cx - 70, cy - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_color, 2)
    else:
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 270, 360, arrow_color, 3)
        end_angle = math.radians(270)
        tip_x = int(cx + radius * math.cos(end_angle))
        tip_y = int(cy + radius * math.sin(end_angle))
        cv2.arrowedLine(frame, (tip_x, tip_y), (tip_x + 20, tip_y - 20), arrow_color, 3, cv2.LINE_AA)
        cv2.putText(frame, f"ROTATE LEFT {abs(angle_diff):.0f}°", (cx - 70, cy - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_color, 2)


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

    ov = frame.copy()
    cv2.fillPoly(ov, [p], fill_color)
    cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

    for i in range(4):
        cv2.line(frame, tuple(p[i]), tuple(p[(i+1)%4]), border_color, 2, cv2.LINE_AA)

    dirs = [(1,1), (-1,1), (-1,-1), (1,-1)]
    for i, (sx, sy) in enumerate(dirs):
        x, y = int(p[i][0]), int(p[i][1])
        L = 20
        cv2.line(frame, (x, y), (x + sx * L, y), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + sy * L), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, marker_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)

    cx = int(np.mean(p[:, 0]))
    cy = int(np.mean(p[:, 1]))
    cv2.drawMarker(frame, (cx, cy), marker_color, cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 15, marker_color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, marker_color, -1, cv2.LINE_AA)
    
    if orientation_info and not orientation_info.get('is_angle_ok', True):
        draw_rotation_arrow(frame, (cx, cy), 
                           orientation_info.get('rotation_direction', 'clockwise'),
                           orientation_info.get('rotation_amount', 0))
    
    if last_data:
        if match_info and match_info.get('is_match', False):
            if orientation_info and not orientation_info.get('is_angle_ok', True):
                label = f"⚠ ROTATE {orientation_info.get('rotation_direction', '').upper()} {orientation_info.get('rotation_amount', 0):.0f}° ⚠"
                label_color = (0, 165, 255)
            else:
                label = f"✓ MATCHED: {last_data[:20]}"
                label_color = (0, 255, 0)
        else:
            label = f"QR: {last_data[:25]}"
            label_color = (0, 0, 255)
        
        tx = max(min(p[0][0], fw - 350), 6)
        ty = max(p[0][1] - 14, 65)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.rectangle(frame, (tx - 5, ty - th - 6), (tx + tw + 5, ty + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (tx - 5, ty - th - 6), (tx + tw + 5, ty + 5), label_color, 1)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 2, cv2.LINE_AA)


def load_seed_image_dialog():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Select Seed QR Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path if file_path else None


# ═══════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════
def main():
    print("═"*60)
    print("  ISRO IROUC 2026 — QR Drone Detector")
    print("  ✓ External webcam support")
    print("  ✓ Long-range QR detection")
    print("  ✓ Auto-calibration")
    print("  ✓ Drone rotation commands")
    print("  ✓ Terminal rotation instructions")
    print("═"*60 + "\n")
    
    # Let user select camera
    print("📷 Available camera indices:")
    print("   Index 0 - Built-in laptop camera")
    print("   Index 1 - External USB webcam")
    print("   Index 2 - Second external camera")
    
    camera_choice = input("\nSelect camera index (default 1 for external webcam): ").strip()
    camera_index = int(camera_choice) if camera_choice else 1
    
    seed_matcher = SeedMatcher()
    drone = DroneController()
    
    # Load seed image
    print("\n📷 Please select a seed QR image...")
    seed_path = load_seed_image_dialog()
    
    if seed_path:
        success, message = seed_matcher.load_seed_from_file(seed_path)
        if success:
            print(f"✅ {message}")
            if seed_matcher.seed_features:
                print(f"   Reference orientation: {seed_matcher.seed_features['angle']:.1f}°")
        else:
            print(f"❌ {message}")
    else:
        print("⚠️  No seed loaded. Running without matching...")
    
    # Connect to drone (optional)
    print("\n🔌 Connect to drone? (y/n): ", end="")
    if input().lower() == 'y':
        drone_ip = input("   Enter drone IP (default 192.168.1.100): ") or "192.168.1.100"
        drone.drone_ip = drone_ip
        if drone.connect():
            drone.control_enabled = True
            print("✅ Drone control enabled - will auto-rotate to match QR")
        else:
            print("❌ Drone connection failed - continuing without drone")
    
    print("\n🎥 Starting webcam...")
    print("\n💡 TIPS FOR BEST DETECTION:")
    print("   - Hold QR code facing the camera")
    print("   - Ensure good lighting")
    print("   - Avoid extreme angles (>45°)")
    print("   - System auto-calibrates after 5 detections\n")
    
    # Initialize webcam with selected index
    try:
        cam = WebcamStream(camera_index)
    except Exception as e:
        print(f"\n❌ Failed to open camera {camera_index}")
        print("   Trying camera index 0 as fallback...")
        cam = WebcamStream(0)
    
    worker = QRWorker(seed_matcher)
    tracker = SimpleTracker()
    
    last_data = ""
    last_match_info = None
    last_orientation = None
    lost_frames = 0
    display_pts = None
    LOSE_AFTER = 30
    no_qr_count = 0
    detection_count = 0
    
    print("\n✅ Ready! Point your webcam at a QR code...\n")
    
    while True:
        ret, frame, cam_fps = cam.read()
        if not ret or frame is None:
            print("⚠️  Webcam error - trying to reconnect...")
            time.sleep(1)
            continue
        
        fh, fw = frame.shape[:2]
        
        worker.submit(frame)
        data, raw_pts, match_info, orientation_info = worker.get()
        
        if match_info and match_info.get('is_match', False):
            last_match_info = match_info
            last_orientation = orientation_info
            no_qr_count = 0
            detection_count += 1
            
            if raw_pts is not None and not seed_matcher.is_calibrated:
                if seed_matcher.auto_calibrate(raw_pts):
                    pass
        
        if raw_pts is not None:
            ordered = order_points(raw_pts.astype(np.float32))
            if is_valid_qr(ordered, fw, fh):
                display_pts = tracker.update(ordered)
                lost_frames = 0
                no_qr_count = 0
                
                if drone.control_enabled and display_pts is not None:
                    cx = int(np.mean(display_pts[:, 0]))
                    cy = int(np.mean(display_pts[:, 1]))
                    drone.navigate_to_qr((cx, cy), (fw//2, fh//2), orientation_info)
                    drone.search_mode = False
                
                if data and data != last_data:
                    if match_info and match_info.get('is_match', False):
                        print(f"\n✅ QR MATCHED! [#{detection_count}]")
                        print(f"   Data: {data}")
                        if orientation_info:
                            print(f"   Angle: {orientation_info['current_angle']:.1f}° (target: {orientation_info['target_angle']:.1f}°)")
                    else:
                        print(f"\n📱 QR Detected: {data}")
                    last_data = data
        else:
            lost_frames += 1
            no_qr_count += 1
            
            if drone.control_enabled and no_qr_count > 10:
                drone.search_for_qr()
                drone.search_mode = True
            
            if lost_frames >= LOSE_AFTER:
                if display_pts is not None:
                    print("\n⚠️  QR lost - searching...")
                display_pts = None
                last_data = ""
                last_match_info = None
                last_orientation = None
                tracker.reset()
        
        # Draw UI
        if display_pts is not None:
            draw_overlay(frame, display_pts, last_data, last_match_info, last_orientation)
            
            if last_match_info and last_match_info.get('is_match', False):
                if last_orientation and not last_orientation.get('is_angle_ok', True):
                    status = "⚠️  ADJUSTING ROTATION"
                    color = (0, 165, 255)
                else:
                    status = "✅ QR MATCHED"
                    color = (0, 255, 0)
            else:
                status = "QR LOCKED"
                color = (0, 0, 255)
        else:
            if drone.control_enabled and drone.search_mode:
                status = "🔍 SEARCHING FOR QR..."
                color = (0, 165, 255)
            else:
                status = "SCANNING..."
                color = (160, 160, 160)
        
        cv2.rectangle(frame, (0, 0), (fw, 50), (0, 0, 0), -1)
        cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        if seed_matcher.seed_loaded:
            seed_short = seed_matcher.seed_data[:25] + "..."
            cv2.putText(frame, f"SEED: {seed_short}", (fw - 280, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 100), 1, cv2.LINE_AA)
            
            if last_orientation:
                angle_diff = last_orientation.get('angle_diff', 0)
                if abs(angle_diff) > 5:
                    cv2.putText(frame, f"ANGLE ERROR: {abs(angle_diff):.1f}°", (fw - 280, 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
        
        if drone.control_enabled:
            cv2.putText(frame, "DRONE: ACTIVE", (fw - 150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"{cam_fps:.0f} FPS", (fw - 80, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q:Quit  L:Load  R:Reset  D:Drone", (12, fh - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.imshow("ISRO IROUC — QR Drone Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if drone.control_enabled:
                drone.land()
            break
        elif key == ord('l'):
            seed_path = load_seed_image_dialog()
            if seed_path:
                success, message = seed_matcher.load_seed_from_file(seed_path)
                if success:
                    print(f"✅ {message}")
                    lost_frames = LOSE_AFTER + 1
                    last_match_info = None
                    last_orientation = None
                else:
                    print(f"❌ {message}")
        elif key == ord('r'):
            seed_matcher.reset()
            last_match_info = None
            last_orientation = None
            detection_count = 0
            print("\n🔄 Seed matcher reset\n")
        elif key == ord('d'):
            drone.control_enabled = not drone.control_enabled
            if drone.control_enabled:
                drone.search_mode = True
                print("\n🚁 Drone control ENABLED - auto-rotation active\n")
            else:
                print("\n🚁 Drone control DISABLED\n")
    
    worker.stop()
    cam.stop()
    drone.disconnect()
    cv2.destroyAllWindows()
    print("\n👋 Detector closed.")


if __name__ == "__main__":
    main()