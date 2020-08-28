import speech_recognition as sr
import os
import playsound
import pyttsx3
from gtts import gTTS

r = sr.Recognizer()


def get_audio():
    with sr.Microphone(sample_rate=48000, chunk_size=2048) as source:
        r.adjust_for_ambient_noise(source)
        playsound.playsound(f'Asset/Audio/start_sound.mp3')
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return text



def put_audio(decoded_translation):
    engine = pyttsx3.init()
    engine.setProperty('rate', 120)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(decoded_translation)
    return engine.runAndWait()