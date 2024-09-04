import os
from datetime import datetime
import pocketsphinx
import pyttsx3
import speech_recognition as sr



rate = 100
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', rate + 50)

name = ""



def speak(audio):
    engine.say(audio)
    print(audio)
    engine.runAndWait()


def me():
    global name
    name = input("What is your name:")


def takeCommand():
    global name
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query.lower() if query else ""
        except sr.WaitTimeoutError:
            speak("Why don't you say something, sir?")
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
        except sr.UnknownValueError:
            speak("Sorry, I didn't understand that.")
            return ""


def greet():
    global name
    hour = datetime.now().hour

    if 0 <= hour < 12:
        speak(f"Good morning! {name}")
        speak("how can i help you?")
    elif 12 <= hour < 18:
        speak(f"Good afternoon! {name}")
        speak("how can i help you?")
    else:
        speak(f"Good evening! {name}")
        speak("how can i help you?")


me()
if __name__ == "__main__":
    greet()
while 1:
    query = takeCommand().lower()

    if "open command prompt" in query:
        os.system("start cmd")
    elif "close command prompt" in query:
        os.system("stop cmd")
    elif "play music" or "play song" in query:
        music_dir = r"C:\Users\user\Music"
        songs = os.listdir(music_dir)
        speak(f"{name} which song to you want me to play")
        takeCommand()
        if takeCommand() == "my favorite song":
            myvas = r"C:\Users\user\Music\y2mate.com - Ruger  Asiwaju Official Audio.mp3"
            os.startfile(os.path.join(music_dir, myvas))
        else:
            rd = random.choice(songs)
            os.startfile(os.path.join(music_dir, rd))
    elif "my ip address" in query:
        ip = get('https://api.ipify.org').text
        speak(f"your  ip address is {ip}")
    elif "discuss with me" or "lets talk" or "conversation with me" in query:
        speak(f"{name}, what do you want us to talk about")
        discuss=takeCommand().lower()
        if "beautiful girls" or "love story" or "about love" in discuss:
            speak(f"{name} do you need to find love?")
            love_wish=takeCommand().lower()
            if "yes" in love_wish:
                speak("to find a true love you need to be yourself and trust in capabilities also never forget that love comes at the right time!")
                speak(f"did i help you {name}")
                response=takeCommand().lower()
                if "yes" in response:
                    speak("grad to here it my friend")