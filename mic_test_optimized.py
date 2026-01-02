import speech_recognition as sr
import sys

def test_microphone():
    """Test microphone and speech recognition with proper error handling."""
    print("🎤 Microphone Test Started")
    
    try:
        r = sr.Recognizer()
        mic = sr.Microphone()
        
        print("✓ Microphone initialized")
        
        with mic as source:
            print("🔊 Calibrating microphone for ambient noise (2 seconds)...")
            r.adjust_for_ambient_noise(source, duration=2)
            print("✓ Calibration complete")
            
            print("\n🎙️ Speak something (listening for 10 seconds)...")
            try:
                audio = r.listen(source, timeout=10, phrase_time_limit=10)
                print("✓ Audio captured")
                
                print("🔍 Processing speech recognition...")
                text = r.recognize_google(audio)
                print(f"\n✅ You said: '{text}'")
                return True
                
            except sr.UnknownValueError:
                print("❌ Could not understand audio. Please speak clearly.")
                return False
            except sr.RequestError as e:
                print(f"❌ Speech API error: {e}")
                print("   Make sure you have an internet connection.")
                return False
            except sr.Timeout:
                print("❌ Listening timed out. Please try again.")
                return False
                
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        print("   Check if microphone is connected and permissions are granted.")
        return False

if __name__ == "__main__":
    success = test_microphone()
    sys.exit(0 if success else 1)
