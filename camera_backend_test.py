import cv2
import numpy as np

print("Testing different camera backends...\n")

# List of backends to try
backends = [
    (cv2.CAP_DSHOW, "DirectShow (Windows)"),
    (cv2.CAP_MSMF, "Media Foundation (Windows)"),
    (cv2.CAP_ANY, "Auto-detect"),
]

for backend_id, backend_name in backends:
    print(f"\n{'='*60}")
    print(f"Testing: {backend_name}")
    print('='*60)
    
    cap = cv2.VideoCapture(0, backend_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera with {backend_name}")
        continue
    
    print(f"✅ Camera opened with {backend_name}")
    print(f"   Backend: {cap.getBackendName()}")
    
    # Try to read a frame
    print("   Reading frame...")
    ret, frame = cap.read()
    
    if not ret:
        print(f"   ❌ Failed to read frame")
        cap.release()
        continue
    
    print(f"   ✅ Frame read successful")
    print(f"   Frame shape: {frame.shape}")
    
    # Check if frame is actually black (all zeros) or just dark
    mean_val = np.mean(frame)
    min_val = np.min(frame)
    max_val = np.max(frame)
    
    print(f"   Pixel statistics:")
    print(f"     Mean: {mean_val:.2f}")
    print(f"     Min: {min_val}")
    print(f"     Max: {max_val}")
    
    if max_val == 0:
        print(f"   ⚠️  COMPLETELY BLACK - All pixels are 0")
    elif mean_val < 10:
        print(f"   ⚠️  VERY DARK - Mean brightness < 10")
    elif mean_val < 50:
        print(f"   ⚠️  DARK - Mean brightness < 50")
    else:
        print(f"   ✅ GOOD BRIGHTNESS - This backend works!")
        
        # Show the frame
        cv2.imshow(f'Test: {backend_name} - Press any key', frame)
        print(f"   Displaying frame... press any key to continue")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cap.release()

print("\n" + "="*60)
print("Test complete!")
print("\nRecommendation: Use the backend that showed GOOD BRIGHTNESS")


