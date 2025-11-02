import cv2
import numpy as np

print("Simple Camera Diagnostic")
print("="*50)

# Try MSMF (Media Foundation) - often works better on Windows
backends_to_try = [
    (cv2.CAP_MSMF, "Media Foundation"),
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_ANY, "Default"),
]

for backend, name in backends_to_try:
    print(f"\nTrying {name}...")
    cap = cv2.VideoCapture(0, backend)
    
    if cap.isOpened():
        # Read multiple frames
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                brightness = np.mean(frame)
                max_val = np.max(frame)
                print(f"  Frame {i+1}: brightness={brightness:.1f}, max_pixel={max_val}")
                
                if max_val > 0 and brightness > 20:
                    print(f"\n✅ SUCCESS! {name} backend works!")
                    print(f"   Use: cv2.VideoCapture(0, cv2.CAP_{backend})")
                    
                    # Show the frame
                    cv2.imshow(f'Working Camera - {name}', frame)
                    print("\nShowing camera feed. Press 'q' to quit.")
                    
                    while True:
                        ret, frame = cap.read()
                        if ret:
                            cv2.putText(frame, f"Backend: {name}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow(f'Working Camera - {name}', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
        
        cap.release()
        print(f"  ❌ {name} produced black/very dark frames")

print("\n❌ No backend produced visible frames.")
print("\nPossible issues:")
print("1. Another app is using the camera (close Teams, Zoom, etc.)")
print("2. Camera privacy settings are blocking desktop apps")
print("3. Camera driver needs update")
print("4. Physical camera cover/obstruction")


