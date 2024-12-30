from langchain_ollama import OllamaLLM
from stt import recognize
from tts import speak

model = OllamaLLM(model="llama3.2")

while True:
    print("Say something (or 'quit' to exit):")
    recognized_text = recognize()

    if recognized_text:
        print(f"You said: {recognized_text}")
        if "quit" in recognized_text or "exit" in recognized_text:
            print("Goodbye!")
            break

        answer = model.invoke(input=recognized_text)
        print(answer)
        speak(answer)
    else:
        print("No valid input detected. Please try again.")