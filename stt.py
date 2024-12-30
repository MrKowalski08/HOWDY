import speech_recognition as sr
import logging

def recognize():
    recognizer = sr.Recognizer()
    logging.basicConfig(level=logging.INFO)

    with sr.Microphone() as mic:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(mic, duration=1)
        audio = recognizer.listen(mic)

        try:
            text = recognizer.recognize_google(audio).lower()
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you repeat?")
            return None
        except sr.RequestError as e:
            print(f"API unavailable or unresponsive: {e}")
            return None