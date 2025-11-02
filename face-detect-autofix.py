import cv2
import time
from collections import deque
import numpy as np

def find_working_camera():
    """Find a camera backend that actually produces visible frames"""
    backends = [
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_ANY, "Default"),
    ]
    
    print("Searching for working camera backend...")
    
    for backend_id, backend_name in backends:
        print(f"  Trying {backend_name}...", end=" ")
        cap = cv2.VideoCapture(0, backend_id)
        
        if not cap.isOpened():
            print("❌ Failed to open")
            continue
        
        # Try reading a few frames
        working = False
        for attempt in range(3):
            ret, frame = cap.read()
            if ret and np.max(frame) > 10:  # Frame has actual content
                brightness = np.mean(frame)
                if brightness > 15:  # Reasonable brightness
                    print(f"✅ Works! (brightness: {brightness:.1f})")
                    return cap, backend_name
        
        print("❌ Produces black frames")
        cap.release()
    
    return None, None

# Load cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Find working camera
cap, backend = find_working_camera()

if cap is None:
    print("\n❌ ERROR: Could not find a working camera backend!")
    print("\nTroubleshooting:")
    print("1. Close ALL apps that might use camera (Teams, Zoom, Skype, Discord)")
    print("2. Check Windows Settings → Privacy → Camera")
    print("   - Enable 'Let apps access your camera'")
    print("   - Enable 'Let desktop apps access your camera'")
    print("3. Make sure camera is not physically blocked")
    print("4. Try unplugging and replugging USB camera (if external)")
    print("5. Restart your computer")
    exit(1)

print(f"\n✅ Using {backend} backend")

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cap.set(cv2.CAP_PROP_CONTRAST, 128)

# Warm up camera
print("Warming up camera...")
for i in range(10):
    ret, _ = cap.read()
time.sleep(0.5)

# Movement tracking variables
position_history = deque(maxlen=5)
time_history = deque(maxlen=5)
smoothed_center = None
frame_count = 0

# Thresholds
MOVEMENT_THRESHOLD = 8
DIRECTION_THRESHOLD = 15
MIN_FACE_SIZE = (80, 80)
SMOOTHING_FACTOR = 0.7

def detect_face_with_eyes(frame, gray):
    """Enhanced face detection"""
    faces1 = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=MIN_FACE_SIZE, flags=cv2.CASCADE_SCALE_IMAGE)
    faces2 = face_cascade_alt.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=MIN_FACE_SIZE)
    all_faces = list(faces1) + list(faces2)
    
    if len(all_faces) == 0:
        return None, 0
    
    best_face = None
    best_score = 0
    
    for (x, y, w, h) in all_faces:
        score = w * h
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        
        if len(eyes) >= 2:
            score *= 1.5
        elif len(eyes) == 1:
            score *= 1.2
        
        frame_center_x = frame.shape[1] / 2
        face_center_x = x + w / 2
        distance_from_center = abs(face_center_x - frame_center_x)
        center_bonus = max(0, 1 - distance_from_center / frame_center_x)
        score *= (1 + center_bonus * 0.3)
        
        if score > best_score:
            best_score = score
            best_face = (x, y, w, h)
    
    confidence = min(100, int((best_score / 50000) * 100))
    return best_face, confidence

def smooth_position(new_center, prev_smoothed):
    """Apply exponential smoothing"""
    if prev_smoothed is None:
        return new_center
    smoothed_x = int(SMOOTHING_FACTOR * new_center[0] + (1 - SMOOTHING_FACTOR) * prev_smoothed[0])
    smoothed_y = int(SMOOTHING_FACTOR * new_center[1] + (1 - SMOOTHING_FACTOR) * prev_smoothed[1])
    return (smoothed_x, smoothed_y)

def calculate_movement_direction(positions, times):
    """Calculate movement direction"""
    if len(positions) < 2:
        return None, None, 0, 0
    
    total_dx = 0
    total_dy = 0
    total_time = 0
    
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        dt = times[i] - times[i-1]
        total_dx += dx
        total_dy += dy
        total_time += dt
    
    avg_dx = total_dx / (len(positions) - 1)
    avg_dy = total_dy / (len(positions) - 1)
    distance = (avg_dx**2 + avg_dy**2) ** 0.5
    speed = distance / total_time if total_time > 0 else 0
    
    direction = None
    abs_dx = abs(avg_dx)
    abs_dy = abs(avg_dy)
    
    if abs_dx > MOVEMENT_THRESHOLD or abs_dy > MOVEMENT_THRESHOLD:
        if abs_dx > abs_dy * 0.7:
            if avg_dx > DIRECTION_THRESHOLD:
                direction = "RIGHT"
            elif avg_dx < -DIRECTION_THRESHOLD:
                direction = "LEFT"
        
        if abs_dy > abs_dx * 0.7:
            if avg_dy > DIRECTION_THRESHOLD:
                direction = "DOWN"
            elif avg_dy < -DIRECTION_THRESHOLD:
                direction = "UP"
        
        if abs_dx > DIRECTION_THRESHOLD and abs_dy > DIRECTION_THRESHOLD:
            h_dir = "RIGHT" if avg_dx > 0 else "LEFT"
            v_dir = "DOWN" if avg_dy > 0 else "UP"
            direction = f"{v_dir}-{h_dir}"
    
    return direction, speed, avg_dx, avg_dy

print("\n" + "="*60)
print("Enhanced Face Movement Tracker Started!")
print(f"Backend: {backend}")
print("Press 'q' to quit")
print("="*60 + "\n")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Auto-boost dark frames
    frame_brightness = np.mean(frame)
    if 0 < frame_brightness < 40:
        frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=40)
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    face, confidence = detect_face_with_eyes(frame, gray)
    current_time = time.time()
    frame_count += 1
    
    if face is not None:
        x, y, w, h = face
        cx = x + w // 2
        cy = y + h // 2
        
        smoothed_center = smooth_position((cx, cy), smoothed_center)
        sx, sy = smoothed_center
        
        position_history.append(smoothed_center)
        time_history.append(current_time)
        
        color = (0, int(confidence * 2.55), int(255 - confidence * 2.55))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (sx, sy), 6, (0, 255, 0), -1)
        cv2.circle(frame, (sx, sy), 3, (255, 255, 255), -1)
        cv2.putText(frame, f"Confidence: {confidence}%", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        direction, speed, dx, dy = calculate_movement_direction(list(position_history), list(time_history))
        
        y_offset = 30
        if direction:
            text = f"Moving: {direction}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            y_offset += 35
            
            speed_text = f"Speed: {speed:.1f} px/s"
            cv2.putText(frame, speed_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
            
            disp_text = f"dx: {dx:.1f}, dy: {dy:.1f}"
            cv2.putText(frame, disp_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            arrow_start = (sx, sy)
            arrow_end = (int(sx + dx * 3), int(sy + dy * 3))
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.3)
        else:
            cv2.putText(frame, "STABLE", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        coord_text = f"Position: ({sx}, {sy})"
        cv2.putText(frame, coord_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        position_history.clear()
        time_history.clear()
        smoothed_center = None
        
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, "Move closer or adjust lighting", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    if frame_count > 1:
        elapsed = current_time - time_history[0] if len(time_history) > 0 else 0.033
        fps = len(time_history) / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow('Enhanced Face Movement Tracker', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Face Movement Tracker Stopped")


