import pyttsx3
import speech_recognition as sr
import time

r = sr.Recognizer()
keywords = [("jarvis", 1),("hey jarvis", 1)]
source = sr.Microphone()

def speak(text):
    rate = 100
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(text)
    engine.runAndWait()
def callback(recognizer, audio):
    try:
        speech_as_text = recognizer.recognize_sphinx(audio, keyword_entries=keywords)
        print(speech_as_text)
        if "jordan" in speech_as_text or "hey jordan":
            speak("yes sir")
            recognize_main()
    except sr.UnknownValueError:
        speak("Oops! could not understand that!")
def start_recognizer():
    print("Listening...")
    r.listen_in_background(source, callback)
    time.sleep(1000000)
def recognize_main():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak("why dont you say any thing")
        audio = r.listen(source)
    data = ""
    try:
        data = r.recognize_google(audio)
        data.lower()
        print("you said: " + data)

        if "how are you" in data:
            speak("I am fine sir")
        elif "hello" in data:
            speak("hello there")
        else:
            speak("sorry sir, i didn't get your request")
    except sr.UnknownValueError:
        speak("Oops! could not understand")
    except sr.RequestError as e:
        print("use please a nice english; {0}".format(e))
if __name__ == "__main__":

 while 1:
    start_recognizer()