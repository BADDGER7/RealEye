import cv2
import sys

def test_camera():
    """Test camera functionality with proper error handling and frame rate display."""
    print("🎥 Camera Test Started")
    print("Press 'ESC' or 'q' to exit")
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access camera. Check if camera is connected.")
            return False
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Camera opened successfully")
        print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Add frame counter and info on display
            cv2.putText(frame, f"Frames: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Camera Test", frame)
            
            # Exit on ESC (27) or 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print(f"\n✓ Test completed. Captured {frame_count} frames")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"❌ Error during camera test: {e}")
        return False

if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)
