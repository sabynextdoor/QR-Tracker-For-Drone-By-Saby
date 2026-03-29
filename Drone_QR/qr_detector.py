"""
ISRO IROUC 2026 — QR Drone Detector with Seed Image Matching
Real FPS display + aggressive tracking + QR pattern matching
"""

import cv2
import numpy as np
import threading
import time
import os
from collections import deque
from tkinter import filedialog, Tk
from tkinter import messagebox
import hashlib


# ═══════════════════════════════════════════════════════════════
#  SEED IMAGE MATCHER
# ═══════════════════════════════════════════════════════════════
class SeedMatcher:
    def __init__(self):
        self.seed_image = None
        self.seed_hash = None
        self.seed_data = None
        self.seed_loaded = False
        self.matching_threshold = 85  # percentage
        
    def load_seed_from_file(self, filepath):
        """Load seed image from file"""
        try:
            # Read image
            img = cv2.imread(filepath)
            if img is None:
                return False, "Could not read image file"
            
            # Store the image
            self.seed_image = img.copy()
            
            # Try to decode QR code from seed image
            detector = cv2.QRCodeDetector()
            data, pts, _ = detector.detectAndDecode(img)
            
            if data and len(data) > 0:
                self.seed_data = data
                # Create hash for fast comparison
                self.seed_hash = hashlib.sha256(data.encode()).hexdigest()
                self.seed_loaded = True
                return True, f"Loaded: {data[:50]}..."
            else:
                return False, "No QR code found in seed image"
                
        except Exception as e:
            return False, f"Error loading seed: {str(e)}"
    
    def compare_with_seed(self, qr_data):
        """Compare detected QR data with seed"""
        if not self.seed_loaded or not qr_data:
            return False, 0
        
        # Exact match
        if qr_data == self.seed_data:
            return True, 100
        
        # Calculate similarity percentage (simple string matching)
        similarity = self._calculate_similarity(qr_data, self.seed_data)
        
        # Check if above threshold
        if similarity >= self.matching_threshold:
            return True, similarity
        
        return False, similarity
    
    def _calculate_similarity(self, str1, str2):
        """Calculate similarity between two strings"""
        if not str1 or not str2:
            return 0
        
        # Use Levenshtein distance concept
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 100
        
        # Simple character matching
        matches = 0
        min_len = min(len(str1), len(str2))
        for i in range(min_len):
            if str1[i] == str2[i]:
                matches += 1
        
        # Calculate percentage
        similarity = (matches / max_len) * 100
        return similarity
    
    def get_seed_info(self):
        """Get information about loaded seed"""
        if self.seed_loaded:
            return {
                'loaded': True,
                'data': self.seed_data,
                'hash': self.seed_hash[:16] + "..."
            }
        return {'loaded': False}
    
    def reset(self):
        """Reset seed matcher"""
        self.seed_image = None
        self.seed_hash = None
        self.seed_data = None
        self.seed_loaded = False


# ═══════════════════════════════════════════════════════════════
#  THREADED CAMERA
# ═══════════════════════════════════════════════════════════════
class CameraStream:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS,          60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)
        self.ret, self.frame = self.cap.read()
        self._lock    = threading.Lock()
        self.running  = True
        self.cam_fps  = 0.0
        self._ftimes  = deque(maxlen=30)
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
#  QR WORKER — 6 strategies, always trying
# ═══════════════════════════════════════════════════════════════
class QRWorker:
    def __init__(self, seed_matcher=None):
        self.detector = cv2.QRCodeDetector()
        self.seed_matcher = seed_matcher
        self._lock    = threading.Lock()
        self._frame   = None
        self._result  = (None, None, None)  # (data, pts, match_info)
        self.running  = True
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
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        k_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        
        while self.running:
            with self._lock:
                frame = self._frame
            if frame is None:
                time.sleep(0.001)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = (None, None)
            match_info = None

            # 1 plain gray
            data, pts = self._try(gray)
            if pts is not None:
                result = (data, pts)

            # 2 CLAHE
            if result[1] is None:
                data, pts = self._try(clahe.apply(gray))
                if pts is not None: 
                    result = (data, pts)

            # 3 sharpen
            if result[1] is None:
                data, pts = self._try(cv2.filter2D(gray, -1, k_sharp))
                if pts is not None: 
                    result = (data, pts)

            # 4 otsu
            if result[1] is None:
                _, th = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                data, pts = self._try(th)
                if pts is not None: 
                    result = (data, pts)

            # 5 downscale 0.6x
            if result[1] is None:
                s = cv2.resize(gray, None, fx=0.6, fy=0.6)
                d, p = self._try(s)
                if p is not None: 
                    result = (d, p / 0.6)

            # 6 upscale 1.8x
            if result[1] is None:
                s = cv2.resize(gray, None, fx=1.8, fy=1.8)
                d, p = self._try(s)
                if p is not None: 
                    result = (d, p / 1.8)

            # Check if QR matches seed
            if result[1] is not None and self.seed_matcher and result[0]:
                is_match, similarity = self.seed_matcher.compare_with_seed(result[0])
                if is_match:
                    match_info = {
                        'is_match': True,
                        'similarity': similarity,
                        'seed_data': self.seed_matcher.seed_data
                    }
                else:
                    match_info = {
                        'is_match': False,
                        'similarity': similarity,
                        'seed_data': None
                    }

            with self._lock:
                self._result = (result[0], result[1], match_info)
                self._frame  = None

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
#  KALMAN — very aggressive, follows hand instantly
# ═══════════════════════════════════════════════════════════════
class KalmanCorners:
    def __init__(self):
        self.kf    = None
        self.ready = False

    def _make(self, pts):
        kf = cv2.KalmanFilter(16, 8)
        F  = np.eye(16, dtype=np.float32)
        for i in range(8):
            F[i, i+8] = 1.0
        kf.transitionMatrix    = F
        kf.measurementMatrix   = np.eye(8, 16, dtype=np.float32)
        kf.processNoiseCov     = np.eye(16, dtype=np.float32) * 0.5
        kf.measurementNoiseCov = np.eye(8,  dtype=np.float32) * 0.01
        kf.errorCovPost        = np.eye(16, dtype=np.float32)
        kf.statePost           = np.zeros(16, dtype=np.float32)
        kf.statePost[:8]       = pts.flatten().astype(np.float32)
        self.kf    = kf
        self.ready = True

    def update(self, pts):
        flat = pts.flatten().astype(np.float32)
        if not self.ready:
            self._make(pts)
            return pts.copy()
        self.kf.predict()
        return self.kf.correct(flat)[:8].reshape(4, 2)

    def predict_only(self):
        if not self.ready:
            return None
        return self.kf.predict()[:8].reshape(4, 2)

    def reset(self):
        self.kf    = None
        self.ready = False


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def order_points(pts):
    pts  = pts.astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype=np.float32)


def is_valid_qr(pts, fw, fh):
    pts[:, 0] = np.clip(pts[:, 0], 0, fw-1)
    pts[:, 1] = np.clip(pts[:, 1], 0, fh-1)
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


def draw_overlay(frame, pts, last_data, match_info=None):
    p  = pts.astype(np.int32)
    fh, fw = frame.shape[:2]

    # Determine color based on match status
    if match_info and match_info.get('is_match', False):
        border_color = (0, 255, 0)  # Green for matched
        fill_color = (0, 150, 0)
        marker_color = (0, 255, 0)
    elif last_data:
        border_color = (0, 0, 255)  # Red for detected but not matched
        fill_color = (0, 0, 150)
        marker_color = (0, 0, 255)
    else:
        border_color = (100, 100, 100)  # Grey for predicted
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
    dirs = [(1,1),(-1,1),(-1,-1),(1,-1)]
    for i,(sx,sy) in enumerate(dirs):
        x, y = int(p[i][0]), int(p[i][1])
        L = 20
        cv2.line(frame,(x,y),(x+sx*L,y),(255,255,255),3,cv2.LINE_AA)
        cv2.line(frame,(x,y),(x,y+sy*L),(255,255,255),3,cv2.LINE_AA)
        cv2.circle(frame,(x,y),6,marker_color,-1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),9,(255,255,255),1,cv2.LINE_AA)

    # Centre
    cx = int(np.mean(p[:,0]))
    cy = int(np.mean(p[:,1]))
    cv2.drawMarker(frame,(cx,cy),marker_color,cv2.MARKER_CROSS,34,2,cv2.LINE_AA)
    cv2.circle(frame,(cx,cy),17,marker_color,1,cv2.LINE_AA)
    cv2.circle(frame,(cx,cy),4,marker_color,-1,cv2.LINE_AA)

    # Label with match info
    if last_data:
        if match_info and match_info.get('is_match', False):
            label = f"✓ MATCHED: {last_data[:35]}"
            label_color = (0, 255, 0)
        else:
            label = f"QR: {last_data[:35]}"
            label_color = (0, 0, 255)
        
        tx = max(min(p[0][0], fw-250), 6)
        ty = max(p[0][1] - 14, 22)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame,(tx-6,ty-th-8),(tx+tw+6,ty+6),(0,0,0),-1)
        cv2.rectangle(frame,(tx-6,ty-th-8),(tx+tw+6,ty+6),label_color,1)
        cv2.putText(frame, label,(tx,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,label_color,2,cv2.LINE_AA)


def load_seed_image_dialog():
    """Open file dialog to load seed image"""
    root = Tk()
    root.withdraw()  # Hide main window
    
    file_path = filedialog.askopenfilename(
        title="Select Seed QR Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path if file_path else None


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("═"*60)
    print("  ISRO IROUC 2026 — QR Drone Detector with Seed Matching")
    print("  Q to quit | L to load seed image | R to reset seed")
    print("═"*60 + "\n")
    
    # Initialize seed matcher
    seed_matcher = SeedMatcher()
    
    # Try to load seed image on startup
    print("Please select a seed QR image...")
    seed_path = load_seed_image_dialog()
    
    if seed_path:
        success, message = seed_matcher.load_seed_from_file(seed_path)
        if success:
            print(f"✓ Seed loaded successfully!")
            print(f"  Target: {seed_matcher.seed_data[:80]}...")
        else:
            print(f"✗ Failed to load seed: {message}")
            print("  Continuing without seed matching...")
    else:
        print("No seed loaded. Continuing without matching...")
    
    print("\nStarting camera...")
    
    cam    = CameraStream(0)
    worker = QRWorker(seed_matcher)
    kalman = KalmanCorners()

    last_data    = ""
    last_match_info = None
    lost_frames  = 0
    display_pts  = None
    LOSE_AFTER   = 30

    # Real render FPS using rolling window
    render_times = deque(maxlen=30)
    
    # Status variables
    show_help = True
    help_timer = time.time()

    while True:
        t0 = time.time()
        ret, frame, cam_fps = cam.read()
        if not ret or frame is None:
            continue

        fh, fw = frame.shape[:2]

        worker.submit(frame)
        data, raw_pts, match_info = worker.get()

        # Update last_match_info if we have a match
        if match_info and match_info.get('is_match', False):
            last_match_info = match_info

        if raw_pts is not None:
            ordered = order_points(raw_pts.astype(np.float32))
            if is_valid_qr(ordered, fw, fh):
                display_pts = kalman.update(ordered)
                lost_frames = 0
                if data and data != last_data:
                    # Check if it matches seed
                    if match_info and match_info.get('is_match', False):
                        print(f"  [MATCHED]  {data} (Similarity: {match_info['similarity']:.1f}%)")
                    else:
                        print(f"  [QR]  {data}")
                    last_data = data
        else:
            lost_frames += 1
            if lost_frames < LOSE_AFTER and kalman.ready:
                pred = kalman.predict_only()
                if pred is not None:
                    display_pts = pred
            if lost_frames >= LOSE_AFTER:
                display_pts = None
                last_data = ""
                last_match_info = None
                kalman.reset()

        # ── Render ───────────────────────────────────────────────────
        if display_pts is not None:
            draw_overlay(frame, display_pts, last_data, last_match_info)
            
            if last_match_info and last_match_info.get('is_match', False):
                s_txt, s_col = "✓ SEED MATCHED ✓", (0, 255, 0)
            else:
                s_txt, s_col = "QR LOCKED", (0, 0, 255)
        else:
            s_txt, s_col = "Scanning...", (160, 160, 160)

        # Top bar
        cv2.rectangle(frame,(0,0),(fw,70),(0,0,0),-1)
        
        # Status text
        cv2.putText(frame, s_txt,(12,42),
                    cv2.FONT_HERSHEY_SIMPLEX,1.1,s_col,2,cv2.LINE_AA)
        
        # Seed info
        if seed_matcher.seed_loaded:
            seed_short = seed_matcher.seed_data[:35] + "..."
            cv2.putText(frame, f"SEED: {seed_short}",(12,68),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,100),1,cv2.LINE_AA)
        
        # Camera FPS
        cv2.putText(frame, f"CAM: {cam_fps:.0f} fps",(fw-210,42),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,200,100),2,cv2.LINE_AA)
        
        # Help text
        if show_help or (time.time() - help_timer) < 3:
            cv2.putText(frame,"Q:Quit  L:LoadSeed  R:ResetSeed",
                        (fw-350, fh-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.48,(200,200,200),1,cv2.LINE_AA)
        
        # Footer
        cv2.putText(frame,"ISRO IROUC 2026 — QR Drone Detector with Seed Matching",
                    (12,fh-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,(200,200,200),1,cv2.LINE_AA)

        cv2.imshow("ISRO IROUC — QR Drone Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            # Load new seed image
            print("\nLoading new seed image...")
            seed_path = load_seed_image_dialog()
            if seed_path:
                success, message = seed_matcher.load_seed_from_file(seed_path)
                if success:
                    print(f"✓ New seed loaded!")
                    print(f"  Target: {seed_matcher.seed_data[:80]}...")
                    # Reset tracking to force re-detection
                    lost_frames = LOSE_AFTER + 1
                    last_match_info = None
                else:
                    print(f"✗ Failed to load seed: {message}")
            else:
                print("No file selected.")
            show_help = True
            help_timer = time.time()
            
        elif key == ord('r'):
            # Reset seed matcher
            print("\nResetting seed matcher...")
            seed_matcher.reset()
            last_match_info = None
            print("Seed matcher reset. Load new seed with 'L' key.")
            show_help = True
            help_timer = time.time()

    worker.stop()
    cam.stop()
    cv2.destroyAllWindows()
    print("\nDetector closed.")


if __name__ == "__main__":
    main()