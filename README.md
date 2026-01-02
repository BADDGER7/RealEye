# Real Eye - Real-Time Object Detection Assistant

Real Eye is a voice-activated computer vision assistant that uses YOLOv8 for real-time object detection from your webcam, combined with speech recognition for voice commands and text-to-speech feedback.[file:1][file:2][file:3] It identifies objects, people, and faces while providing natural language descriptions of the scene. Designed for accessibility and interactive use on Windows.

## Features
- Real-time YOLOv8 object detection (70+ classes including people, vehicles, animals, electronics).[file:1]
- Voice commands: "detect" to start/stop, "describe" for scene summary, "status", "help", "exit".[file:1]
- Automatic spoken announcements of detected objects with confidence levels.[file:1]
- Face detection with optional known face recognition (add images to `faces/` folder).[file:1]
- Test scripts for camera (`test_cam_optimized.py`) and microphone (`mic_test_optimized.py`) setup.[file:2][file:3]

## Windows 11 Setup
Install Python 3.10+ from python.org and ensure `pip` is updated (`python -m pip install --upgrade pip`).

Run these commands in Command Prompt or PowerShell (as administrator if needed for microphone/camera access):


Grant camera/microphone permissions to Python in Windows Settings > Privacy & security > Camera/Microphone.[file:1][file:2][file:3]

## Quick Start
1. Test hardware first:
2. Run main script:
3. Say "detect" to enable vision, "describe" for details, "stop" to pause.[file:1]

Press ESC or 'q' to quit. Webcam opens at 640x480@30fps by default.[file:1]

## Configuration
Edit constants in `main_fixed_v4.py`:
- `CONFIDENCE_THRESHOLD = 0.45`: Detection sensitivity.[file:1]
- `SPEAK_INTERVAL = 3`: Speech announcement frequency (seconds).[file:1]
- `CAMERA_WIDTH/HEIGHT/FPS`: Adjust video settings.[file:1]

Add known faces: Place JPG/PNG images in `faces/` folder; program auto-loads them.[file:1]

## Voice Commands
| Command | Action |
|---------|--------|
| detect/start | Enable object detection [file:1] |
| stop/off | Disable detection [file:1] |
| describe | Detailed scene description [file:1] |
| status | Current state and counts [file:1] |
| help | List commands [file:1] |
| exit/quit | Shutdown [file:1] |

## Troubleshooting (Windows 11)
- **Camera not opening**: Run as admin, check Device Manager for webcam drivers.[file:2]
- **Mic errors**: Verify internet (Google Speech API), test in Windows Voice Recorder.[file:3]
- **TTS silent**: Install Microsoft Speech Platform voices via Control Panel.[file:1]
- **YOLO slow**: Use CPU by default; for GPU, install CUDA-enabled PyTorch separately.[file:1]
- Logs appear in console for debugging.[file:1]

## License
MIT License - Free to use, modify, distribute. No warranty provided.[file:1][file:2][file:3]
