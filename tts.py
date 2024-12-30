import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
    
    # Get available voices
    voices = engine.getProperty('voices')
    
    # Check if a third voice exists, otherwise fallback to the first voice
    if len(voices) > 2:
        engine.setProperty('voice', voices[2].id)  # Use third voice if available
    else:
        engine.setProperty('voice', voices[0].id)  # Use default voice
    
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    message = "Hello, I'm your greatest creation"
    speak(message)