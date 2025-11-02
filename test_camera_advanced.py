import cv2
import numpy as np

print("Testing camera with brightness adjustments...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Adjust camera settings for better exposure
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto exposure ON
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Try different values: -7 to -1
cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # 0-255, default usually 128
cap.set(cv2.CAP_PROP_CONTRAST, 128)    # 0-255
cap.set(cv2.CAP_PROP_GAIN, 50)         # Increase gain for low light

print("Camera settings applied. Waiting for camera to adjust...")
print("Current settings:")
print(f"  Auto Exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
print(f"  Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
print(f"  Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
print(f"  Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
print(f"  Gain: {cap.get(cv2.CAP_PROP_GAIN)}")

# Discard first few frames to let camera adjust
print("\nDiscarding first 10 frames for camera warm-up...")
for i in range(10):
    ret, frame = cap.read()
    print(f"Frame {i+1}: captured={ret}", end='\r')

print("\n\nCapturing and displaying frames...")
print("Press 'q' to quit")
print("Press 'b' to increase brightness")
print("Press 'd' to decrease brightness")

brightness = 128

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read frame")
        break
    
    # Check if frame is too dark
    mean_brightness = np.mean(frame)
    
    # Automatically boost brightness if too dark
    if mean_brightness < 50:
        frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=50)
        cv2.putText(frame, "AUTO BRIGHTNESS BOOST APPLIED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display frame statistics
    cv2.putText(frame, f"Mean Brightness: {mean_brightness:.1f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Min: {np.min(frame)}, Max: {np.max(frame)}", (10, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Camera Test - Press Q to quit', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('b'):
        brightness = min(255, brightness + 20)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        print(f"Brightness increased to {brightness}")
    elif key == ord('d'):
        brightness = max(0, brightness - 20)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        print(f"Brightness decreased to {brightness}")

cap.release()
cv2.destroyAllWindows()
print("\nTest complete.")
print(f"Final mean brightness was: {mean_brightness:.1f}")
print("If brightness was < 50, your camera/room is very dark.")

