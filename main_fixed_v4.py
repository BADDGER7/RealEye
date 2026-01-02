

import cv2
import time
import threading
import queue
import pyttsx3
import speech_recognition as sr
import numpy as np
import logging
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# ===============================
# LOGGING SETUP
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================
# GLOBAL STATE & THREAD SAFETY
# ===============================
running = False
exit_program = False
cmd_queue = queue.Queue()
speech_queue = queue.Queue()  # Speech queue to prevent overlapping

# Thread locks for safe access to shared data
detection_lock = threading.Lock()
state_lock = threading.Lock()

# Global detection results (protected by detection_lock)
last_detected_objects = []  # Now stores (object_name, confidence)
last_detected_faces = []
last_spoken_objects = []

# ===============================
# CONFIGURATION
# ===============================
CONFIG = {
    'FACE_FOLDER': 'faces',
    'CONFIDENCE_THRESHOLD': 0.45,  # Lower for better person detection
    'FACE_MATCH_THRESHOLD': 0.5,
    'SPEAK_INTERVAL': 3,  # Speak every 3 seconds about detections
    'FRAME_SKIP': 1,  # Process every frame for better detection
    'CAMERA_WIDTH': 640,
    'CAMERA_HEIGHT': 480,
    'CAMERA_FPS': 30,
    'TTS_RATE': 150,
    'LISTEN_TIMEOUT': 5,
    'PHRASE_TIME_LIMIT': 5,
    'API_RETRY_COUNT': 3,
    'API_BACKOFF_SECONDS': 2,
    
    # Object descriptions for natural speech - EXPANDED with 70+ objects!
    'OBJECT_DESCRIPTIONS': {
        # PEOPLE & ANIMALS
        'person': 'a person',
        'dog': 'a dog',
        'cat': 'a cat',
        'horse': 'a horse',
        'sheep': 'a sheep',
        'cow': 'a cow',
        'elephant': 'an elephant',
        'bear': 'a bear',
        'zebra': 'a zebra',
        'giraffe': 'a giraffe',
        'bird': 'a bird',
        
        # VEHICLES
        'bicycle': 'a bicycle',
        'car': 'a car',
        'motorcycle': 'a motorcycle',
        'airplane': 'an airplane',
        'bus': 'a bus',
        'train': 'a train',
        'truck': 'a truck',
        'boat': 'a boat',
        
        # OUTDOOR OBJECTS
        'traffic light': 'a traffic light',
        'fire hydrant': 'a fire hydrant',
        'stop sign': 'a stop sign',
        'parking meter': 'a parking meter',
        'bench': 'a bench',
        'potted plant': 'a potted plant',
        
        # SPORTS & RECREATION
        'baseball bat': 'a baseball bat',
        'baseball glove': 'a baseball glove',
        'skateboard': 'a skateboard',
        'tennis racket': 'a tennis racket',
        'frisbee': 'a frisbee',
        'skis': 'skis',
        'snowboard': 'a snowboard',
        'sports ball': 'a sports ball',
        'kite': 'a kite',
        'surfboard': 'a surfboard',
        
        # FOOD & DRINK
        'apple': 'an apple',
        'banana': 'a banana',
        'orange': 'an orange',
        'sandwich': 'a sandwich',
        'hot dog': 'a hot dog',
        'pizza': 'a pizza',
        'donut': 'a donut',
        'cake': 'a cake',
        'broccoli': 'broccoli',
        'carrot': 'a carrot',
        'bottle': 'a bottle',
        'wine glass': 'a wine glass',
        'cup': 'a cup',
        'bowl': 'a bowl',
        
        # FURNITURE
        'chair': 'a chair',
        'couch': 'a couch',
        'bed': 'a bed',
        'dining table': 'a dining table',
        'toilet': 'a toilet',
        'desk': 'a desk',
        
        # ELECTRONICS & TECH
        'tv': 'a television',
        'laptop': 'a laptop',
        'mouse': 'a mouse',
        'remote': 'a remote',
        'keyboard': 'a keyboard',
        'microwave': 'a microwave',
        'oven': 'an oven',
        'toaster': 'a toaster',
        'sink': 'a sink',
        'refrigerator': 'a refrigerator',
        'book': 'a book',
        'clock': 'a clock',
        'phone': 'a phone',
        'monitor': 'a monitor',
        
        # PERSONAL ITEMS & ACCESSORIES
        'backpack': 'a backpack',
        'umbrella': 'an umbrella',
        'handbag': 'a handbag',
        'tie': 'a tie',
        'suitcase': 'a suitcase',
        'scissors': 'scissors',
        'teddy bear': 'a teddy bear',
        'hair drier': 'a hair drier',
        'toothbrush': 'a toothbrush',
        'vase': 'a vase',
        
        # KITCHEN ITEMS
        'fork': 'a fork',
        'knife': 'a knife',
        'spoon': 'a spoon',
        
        # WRAITH GAME & CUSTOM
        'sword': 'a sword',
        'gun': 'a gun',
    }
}

# ===============================
# ENHANCED SPEECH WITH QUEUE
# ===============================
def speak(text, priority=False):
    """Convert text to speech using queue to prevent overlapping."""
    if not text or not text.strip():
        return
    
    # Add to speech queue with priority option
    speech_queue.put((priority, text))

def speech_worker():
    """Process speech requests from queue (runs in separate thread)."""
    global exit_program
    while not exit_program:
        try:
            # Get speech request from queue
            priority, text = speech_queue.get(timeout=1)
            
            try:
                # Create new engine for each speech
                engine = pyttsx3.init()
                engine.setProperty('rate', CONFIG['TTS_RATE'])
                logger.debug(f"🔊 Speaking: {text}")
                engine.say(str(text))
                engine.runAndWait()
                # NOTE: Removed engine.quit() - it doesn't exist in pyttsx3
                # The engine will be garbage collected when out of scope
                time.sleep(0.3)  # Small delay between speech outputs
            except Exception as e:
                logger.error(f"TTS error: {e}")
                time.sleep(1)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Speech worker error: {e}")
            time.sleep(0.5)

def format_detection(obj_name, confidence):
    """Format object detection for natural speech."""
    obj_name_lower = obj_name.lower()
    
    # Use description if available, otherwise use object name
    descriptions = CONFIG['OBJECT_DESCRIPTIONS']
    description = descriptions.get(obj_name_lower, obj_name_lower)
    
    # Add confidence percentage
    confidence_pct = int(confidence * 100)
    
    if confidence_pct >= 90:
        return f"{description} at {confidence_pct} percent confidence"
    elif confidence_pct >= 75:
        return f"{description} at {confidence_pct} percent confidence"
    else:
        return f"{description} possibly at {confidence_pct} percent"

def generate_detailed_description(objects_list, faces_count):
    """Generate a detailed, natural description of the scene."""
    if not objects_list and faces_count == 0:
        return "I don't see anything right now. The scene appears to be empty."
    
    description_parts = []
    
    # Organize objects by category
    object_counts = defaultdict(int)
    object_confidences = defaultdict(list)
    
    for obj_name, conf in objects_list:
        obj_name_lower = obj_name.lower()
        object_counts[obj_name_lower] += 1
        object_confidences[obj_name_lower].append(conf)
    
    # PEOPLE & ANIMALS
    people_animals = []
    animal_keywords = ['person', 'dog', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bird']
    for obj in animal_keywords:
        if obj in object_counts:
            count = object_counts[obj]
            avg_conf = sum(object_confidences[obj]) / len(object_confidences[obj])
            descriptions = CONFIG['OBJECT_DESCRIPTIONS']
            desc = descriptions.get(obj, obj)
            
            if count > 1:
                people_animals.append(f"{count} {obj}s (at {int(avg_conf*100)}% confidence)")
            else:
                people_animals.append(f"one {desc} (at {int(avg_conf*100)}% confidence)")
    
    # VEHICLES
    vehicles = []
    vehicle_keywords = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
    for obj in vehicle_keywords:
        if obj in object_counts:
            count = object_counts[obj]
            avg_conf = sum(object_confidences[obj]) / len(object_confidences[obj])
            descriptions = CONFIG['OBJECT_DESCRIPTIONS']
            desc = descriptions.get(obj, obj)
            
            if count > 1:
                vehicles.append(f"{count} {obj}s (at {int(avg_conf*100)}% confidence)")
            else:
                vehicles.append(f"one {desc} (at {int(avg_conf*100)}% confidence)")
    
    # FOOD & ITEMS
    items = []
    for obj_name_lower, count in object_counts.items():
        if obj_name_lower not in animal_keywords and obj_name_lower not in vehicle_keywords:
            avg_conf = sum(object_confidences[obj_name_lower]) / len(object_confidences[obj_name_lower])
            descriptions = CONFIG['OBJECT_DESCRIPTIONS']
            desc = descriptions.get(obj_name_lower, obj_name_lower)
            
            if count > 1:
                items.append(f"{count} {obj_name_lower}s (at {int(avg_conf*100)}% confidence)")
            else:
                items.append(f"one {desc} (at {int(avg_conf*100)}% confidence)")
    
    # Build detailed description
    full_description = "Here is what I see: "
    
    if people_animals:
        full_description += "In the foreground, I can identify " + ", ".join(people_animals) + ". "
    
    if vehicles:
        full_description += "I can also see " + ", ".join(vehicles) + ". "
    
    if items:
        full_description += "Additionally, I detect " + ", ".join(items) + ". "
    
    if faces_count > 0:
        face_text = f"I also detected {faces_count} face" if faces_count == 1 else f"I also detected {faces_count} faces"
        full_description += face_text + ". "
    
    full_description += "That summarizes the current scene in detail."
    
    return full_description

# ===============================
# FACE RECOGNITION SETUP
# ===============================
known_face_encodings = []
known_face_names = []

def get_face_encoding(img):
    """Extract face encoding from image using simple resizing and normalization."""
    try:
        if img is None or img.size == 0:
            return None
        img_resized = cv2.resize(img, (112, 112))
        return img_resized.flatten().astype(np.float32) / 255.0
    except Exception as e:
        logger.warning(f"Face encoding error: {e}")
        return None

def load_known_faces():
    """Load all known faces from faces/ folder."""
    global known_face_encodings, known_face_names
    face_path = Path(CONFIG['FACE_FOLDER'])
    
    if not face_path.exists():
        face_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created {CONFIG['FACE_FOLDER']} folder. Add face images here.")
        return
    
    loaded_count = 0
    for file in face_path.iterdir():
        if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            try:
                img = cv2.imread(str(file))
                if img is not None:
                    encoding = get_face_encoding(img)
                    if encoding is not None:
                        known_face_encodings.append(encoding)
                        known_face_names.append(file.stem)
                        loaded_count += 1
                        logger.info(f"✓ Loaded face: {file.stem}")
            except Exception as e:
                logger.error(f"Error loading {file.name}: {e}")
    
    logger.info(f"Total faces loaded: {loaded_count}")

# ===============================
# IMPROVED SPEECH RECOGNITION
# ===============================
def speech_listener():
    """Listen for voice commands with better error handling."""
    global exit_program
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Calibrate microphone
    try:
        with mic as source:
            logger.info("🎤 Calibrating microphone for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("🎤 Microphone ready!")
    except Exception as e:
        logger.error(f"Microphone initialization error: {e}")
        return
    
    api_error_count = 0
    
    while not exit_program:
        try:
            with mic as source:
                audio = recognizer.listen(
                    source,
                    timeout=CONFIG['LISTEN_TIMEOUT'],
                    phrase_time_limit=CONFIG['PHRASE_TIME_LIMIT']
                )
                
                # Try to recognize with retry logic
                cmd = None
                for attempt in range(CONFIG['API_RETRY_COUNT']):
                    try:
                        cmd = recognizer.recognize_google(audio).lower()
                        api_error_count = 0
                        break
                    except sr.RequestError as e:
                        logger.warning(f"API error (attempt {attempt + 1}): {e}")
                        if attempt < CONFIG['API_RETRY_COUNT'] - 1:
                            time.sleep(CONFIG['API_BACKOFF_SECONDS'])
                        else:
                            api_error_count += 1
                
                if api_error_count >= 3:
                    speak("I'm having trouble connecting to the internet. Please check your connection.", priority=True)
                    api_error_count = 0
                
                if cmd:
                    logger.info(f"🎙️ Command received: {cmd}")
                    cmd_queue.put(cmd)
        
        except sr.UnknownValueError:
            pass  # User spoke but not recognized
        except sr.Timeout:
            pass  # Timeout is normal, keep listening
        except Exception as e:
            logger.error(f"Speech listener error: {e}")
            time.sleep(0.5)

# ===============================
# VISION LOOP WITH DETECTION
# ===============================
def vision_loop():
    """Main vision loop with object and person detection."""
    global running, exit_program, last_detected_objects, last_detected_faces, last_spoken_objects
    
    # Load YOLO model
    try:
        logger.info("🤖 Loading YOLO model...")
        model = YOLO("yolov8n.pt")
        logger.info("✓ YOLO model loaded successfully")
        speak("Vision system loaded", priority=True)
    except Exception as e:
        logger.error(f"Error loading YOLO: {e}")
        speak("Failed to load vision model", priority=True)
        return
    
    # Open camera
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            logger.error("Cannot open camera!")
            speak("Cannot access camera", priority=True)
            return
        
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['CAMERA_WIDTH'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['CAMERA_HEIGHT'])
        cam.set(cv2.CAP_PROP_FPS, CONFIG['CAMERA_FPS'])
        logger.info("✓ Camera opened")
        speak("Camera initialized", priority=True)
    except Exception as e:
        logger.error(f"Camera error: {e}")
        speak("Camera initialization failed", priority=True)
        return
    
    # Load face cascade
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    load_known_faces()
    
    last_speak_time = 0
    frame_count = 0
    detection_history = defaultdict(int)
    
    logger.info("✓ Vision loop started. Say 'detect' to start detection.")
    
    while not exit_program:
        try:
            ret, frame = cam.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.5)
                continue
            
            detected_objects_with_conf = []
            detected_faces = []
            frame_count += 1
            
            # Display frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if running and frame_count % CONFIG['FRAME_SKIP'] == 0:
                
                # ==================
                # YOLO DETECTION
                # ==================
                try:
                    results = model(frame, verbose=False, conf=CONFIG['CONFIDENCE_THRESHOLD'])
                    
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            
                            for box, cls, conf in zip(boxes, classes, confidences):
                                if conf >= CONFIG['CONFIDENCE_THRESHOLD']:
                                    obj_name = model.names[int(cls)]
                                    detected_objects_with_conf.append((obj_name, float(conf)))
                                    
                                    # Draw bounding box
                                    x1, y1, x2, y2 = box
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                                 (0, 255, 0), 2)
                                    
                                    # Draw label
                                    label = f"{obj_name} {conf:.2f}"
                                    cv2.putText(frame, label, (int(x1), int(y1) - 5),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    logger.info(f"✓ Detected: {obj_name} ({conf:.2f})")
                except Exception as e:
                    logger.error(f"YOLO detection error: {e}")
                
                # ==================
                # FACE DETECTION
                # ==================
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        detected_faces.append((x, y, w, h))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, "Face", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # If we have known faces, try to match
                        if known_face_encodings:
                            roi = gray[y:y+h, x:x+w]
                            face_encoding = get_face_encoding(frame[y:y+h, x:x+w])
                            if face_encoding is not None:
                                detected_faces.append(("unknown_face", face_encoding))
                
                except Exception as e:
                    logger.error(f"Face detection error: {e}")
                
                # Update global detection state
                with detection_lock:
                    last_detected_objects = detected_objects_with_conf
                    last_detected_faces = detected_faces
                
                # ==================
                # VOICE FEEDBACK
                # ==================
                current_time = time.time()
                if current_time - last_speak_time >= CONFIG['SPEAK_INTERVAL']:
                    
                    announcements = []
                    
                    # Announce detected objects
                    if detected_objects_with_conf:
                        # Count and group objects
                        object_counts = defaultdict(int)
                        for obj_name, conf in detected_objects_with_conf:
                            object_counts[obj_name] += 1
                        
                        # Create announcement
                        obj_list = []
                        for obj_name, count in object_counts.items():
                            formatted = format_detection(obj_name, 0.85)  # Average confidence
                            if count > 1:
                                obj_list.append(f"{count} {obj_name}s")
                            else:
                                obj_list.append(formatted)
                        
                        if obj_list:
                            announcements.append("I can see: " + ", ".join(obj_list))
                    
                    # Announce detected faces
                    if detected_faces:
                        announcements.append(f"I detected {len(detected_faces)} face{'s' if len(detected_faces) > 1 else ''}")
                    
                    if not announcements:
                        announcements.append("I don't see any objects or people right now")
                    
                    # Speak the announcement
                    for announcement in announcements:
                        speak(announcement)
                    
                    last_speak_time = current_time
            
            # ==================
            # DISPLAY VIDEO
            # ==================
            cv2.imshow("Vision Assistant", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                logger.info("Exiting vision loop...")
                exit_program = True
                break
            elif key == ord('d'):  # 'd' to toggle detection
                running = not running
                state = "ON" if running else "OFF"
                logger.info(f"Detection toggled {state}")
                speak(f"Detection turned {state}")
        
        except Exception as e:
            logger.error(f"Vision loop error: {e}")
            time.sleep(0.5)
    
    # Cleanup
    cam.release()
    cv2.destroyAllWindows()
    logger.info("✓ Vision loop closed")

# ===============================
# COMMAND PROCESSING
# ===============================
def process_commands():
    """Process voice commands."""
    global running, exit_program
    
    commands_info = {
        'detect': 'Enable detection',
        'stop': 'Stop detection',
        'describe': 'Detailed description of what you see',
        'exit': 'Exit program',
        'status': 'Show status',
        'help': 'Show help'
    }
    
    speak("Command processor ready. Say 'help' for available commands.", priority=True)
    
    while not exit_program:
        try:
            cmd = cmd_queue.get(timeout=1)
            
            if 'detect' in cmd or 'start' in cmd:
                if not running:
                    running = True
                    speak("Detection enabled. I'm now looking for objects and people.")
                    logger.info("✓ Detection started")
                else:
                    speak("Detection is already running.")
            
            elif 'stop' in cmd or 'off' in cmd:
                if running:
                    running = False
                    speak("Detection disabled.")
                    logger.info("✓ Detection stopped")
                else:
                    speak("Detection is already stopped.")
            
            elif 'describe' in cmd:
                with detection_lock:
                    objects = last_detected_objects.copy()
                    faces = len(last_detected_faces)
                
                if objects or faces > 0:
                    detailed_desc = generate_detailed_description(objects, faces)
                    speak(detailed_desc, priority=True)
                    logger.info(f"✓ Detailed description provided")
                else:
                    speak("I don't currently see anything to describe. Please enable detection first.", priority=True)
            
            elif 'status' in cmd:
                status = "ON" if running else "OFF"
                with detection_lock:
                    obj_count = len(last_detected_objects)
                    face_count = len(last_detected_faces)
                speak(f"Detection is {status}. I can see {obj_count} objects and {face_count} faces.")
            
            elif 'help' in cmd:
                help_text = "Available commands: detect, stop, describe, status, help, exit"
                speak(help_text)
            
            elif 'exit' in cmd or 'quit' in cmd:
                speak("Shutting down. Goodbye!")
                exit_program = True
            
            else:
                speak(f"Unknown command: {cmd}. Say help for available commands.")
        
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Command processing error: {e}")

# ===============================
# MAIN PROGRAM
# ===============================
def main():
    """Main program entry point."""
    global exit_program
    
    logger.info("=" * 50)
    logger.info("🤖 VISION ASSISTANT STARTING")
    logger.info("=" * 50)
    logger.info(f"📊 Loaded {len(CONFIG['OBJECT_DESCRIPTIONS'])} object descriptions")
    logger.info(f"🎤 Commands: detect, stop, describe, status, help, exit")
    
    speak("Vision assistant starting up", priority=True)
    
    # Create and start threads
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    listener_thread = threading.Thread(target=speech_listener, daemon=True)
    command_thread = threading.Thread(target=process_commands, daemon=True)
    
    try:
        vision_thread.start()
        speech_thread.start()
        listener_thread.start()
        command_thread.start()
        
        logger.info("✓ All threads started")
        
        # Keep main thread alive
        while not exit_program:
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Shutting down...")
        exit_program = True
        
        # Wait for threads to finish
        vision_thread.join(timeout=2)
        speech_thread.join(timeout=2)
        listener_thread.join(timeout=2)
        command_thread.join(timeout=2)
        
        logger.info("=" * 50)
        logger.info("✓ VISION ASSISTANT STOPPED")
        logger.info("=" * 50)

if __name__ == "__main__":
    main()
