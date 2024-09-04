import pyttsx3
import speech_recognition as sr
import datetime
import os
import cv2
import random
from requests import get
import wikipedia
import smtplib
import webbrowser
import pywhatkit as kit
from smtplib import SMTPException
import sys
import time
import pyjokes
import psutil

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)

name = ""


# text to speech
def speak(audio):
    engine.say(audio)
    print(audio)
    engine.runAndWait()


# your name
def me():
    global name
    speak("sir, I would like to know your name?")
    name = input("Name: ")
    return name

#close some applications
def close_program(application):
    os.system(f"taskkill /F /IM {application}.exe")
# speech to text
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
            speak(f"Why dont you say anything {name}")
            audio = r.listen(source, timeout=10, phrase_time_limit=10)
        except sr.UnknownValueError:
            speak("sorry, I didn't understand that")
            speak("Say that again please...")
            return ""


# to wish
def wish():
    global name
    hour = int(datetime.datetime.now().hour)

    if hour >= 0 and hour <= 12:
        speak("Good morning")
        speak("i am DOWKIN. the best AI for your service,")
        name = me()
        speak(f"{name}, what can DOWKIN do to help you?")

    elif hour > 12 and hour < 18:
        speak("Good afternoon")
        speak("i am DOWKIN. the best AI for your service,")
        name = me()
        speak(f"{name}, what can DOWKIN do to help you?")
    else:
        speak("Good evening")
        speak("i am DOWKIN. the best AI for your service,")
        name = me()
        speak(f"{name}, what can DOWKIN do to help you?")


# to send email
def sendEmail(to, content):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login('kagabolucky72@gmail.com', 'kail123#')
        server.sendmail('kagabolucky72@gmail.com', to, content)
        server.close()
        print("Email sent successfully!")
    except SMTPException as e:
        print("SMTPException occured:", e)
    except ConnectionError as e:
        print("ConnectionError occured:", e)
    except Exception as e:
        print("Unexpected error occured:", e)
#to close all programs
def close_all_programs():
    for process in psutil.process_iter():
        try:
            process.terminate()
        except Exception as e:
            speak(f"Failed to terminate process {process.pid}: {e}")
# generate jokes
def get_joke():
    try:
        joke = pyjokes.get_joke()
        return joke
    except pyjokes.JokeException as e:
        print("Error retrieving joke:", e)
        return speak("Sorry, i couldn't get a joke at the moment.")


if __name__ == "__main__":
    wish()
    while True:

        query = takeCommand().lower()

        # logic building for tasks
        # to open notepad
        if "open notepad" in query:
            npath = r"C:\Windows\System32\notepad.exe"
            os.startfile(npath)
        # to open everything
        elif "open everything" in query:
            path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs"
            os.startfile(path)
        # to run cmd
        elif "open command prompt" in query:
            os.system("start cmd")
        # to open camera
        elif "open camera" in query:
            cap = cv2.VideoCapture(0)
            while True:
                ret, img = cap.read()
                cv2.imshow('webcam', img)
                k = cv2.waitKey(50)
                if k == 27:
                    break;
            cap.release()
            cv2.destroyAllWindows()
        # to play music
        elif "play music" in query:
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
        # to see the ip address of your machine
        elif "ip address" in query:
            ip = get('https://api.ipify.org').text
            speak(f"your ip adress  is {ip}")
        # to open wikipedia
        elif "wikipedia" in query:
            speak("searching Wikipedia...")
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("according to wikipedia")
            speak(results)
            # print(results)
        # open youtube
        elif "open youtube" in query:
            webbrowser.open("www.youtube.com")
        # to open slack
        elif "open slack" in query:
            webbrowser.open("www.slack.com")
        # to open stackoverflow
        elif "open stackoverflow" in query:
            webbrowser.open("www.stackoverflow.com")
        # to open google
        elif "open google" in query:
            speak("sir, what should i search on goolge")
            cm = takeCommand().lower()
            webbrowser.open(f"{cm}")
        # to send message via telephone
        elif "send message" in query:
            kit.sendwhatmsg("+250739200728", "this is testing protocol", 2, 25)
        # to play music ony youtube
        elif "play song on youtube" in query:
            kit.playonyt("see you again")
        # to send email
        elif "email to loick" in query:
            try:
                speak("what should i say?")
                content = takeCommand()
                to = "ndahiroloicke@gmail.com"
                sendEmail(to, content)
                speak("Email sent to bedo")

            except Exception as e:
                print("Email enabled to be sent")
        # close application
        elif f"close the apps" in query:
            speak(f"which app do you want me to close {name}")
            app = input("app name")
            #close AI
            if app.lower() == "dowkin":
                speak(f"thanks for using me {name}, have a good day!!")
                sys.exit()
            elif app.lower() == "notepad":
                speak(f"okay {name} closing {app}.")
                os.system("taskkill /F /IM notepad.exe")
            elif app.lower() == "all programs":
                close_all_programs()
                speak("all apps are closed")
            else:
                speak(f"closing the {app}")
                close_program(app)

        # to set alarm
        elif "set alarm" in query:
            nn = int(datetime.datetime.now().hour)
            if nn == 22:
                music_dir = r'C:\Users\user\Music'
                songs = os.listdir(music_dir)
                os.startfile(os.path.join(music_dir, songs[0]))
        # to find a joke
        elif "tell me a joke" in query:
            joke = get_joke()
            speak(joke)
            print(joke)
        # to shut down the machine
        elif "shutdown the system now" in query or "shutdown the computer" in query:
            speak("are you sure you want to shutdown in three seconds")
            if takeCommand() == "yes":
                os.system("shutdown /s /t 3")
            else:
                speak("the shutdown process was terminated")
        # to restart the machine
        elif "restart the laptop" in query or "restart the computer" in query:
            speak("the computer is about to restart in five seconds")
            os.system("shutdown /r /t 5")
        # to make the machine sleep
        elif "sleep the system" in query or "sleep the computer" in query:
            speak("the computer is about to sleep in three seconds")
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        #to open game
        elif "i want to play game" in query or "play nba" in query:
            os.system("nba2k23")
        elif "i want to manage my wifibox" or "i want to change wifi settings" in query:
            ipAd

    else:
        speak("I don't understand")
