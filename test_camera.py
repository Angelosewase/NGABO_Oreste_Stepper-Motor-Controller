import cv2
import time

print("Testing camera access...")

# Try with DirectShow
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"Camera opened: {cap.isOpened()}")

if not cap.isOpened():
    print("Failed to open camera with DSHOW, trying default...")
    cap = cv2.VideoCapture(0)
    print(f"Camera opened: {cap.isOpened()}")

if cap.isOpened():
    print("Camera opened successfully!")
    print(f"Backend: {cap.getBackendName()}")
    print("Attempting to read frame...")
    
    # Set a timeout
    start_time = time.time()
    timeout = 5  # 5 seconds
    
    ret, frame = cap.read()
    elapsed = time.time() - start_time
    
    print(f"Frame read took {elapsed:.2f} seconds")
    print(f"Frame captured: {ret}")
    
    if ret:
        print(f"Frame shape: {frame.shape}")
        print("SUCCESS! Camera is working.")
        cv2.imshow('Test Frame', frame)
        print("Press any key to close...")
        cv2.waitKey(0)
    else:
        print("ERROR: Could not read frame from camera")
else:
    print("ERROR: Could not open camera at all")

cap.release()
cv2.destroyAllWindows()
print("Test complete.")

