# common
import numpy as np
from PIL import Image
import pytesseract
import moviepy.editor as mp
import speech_recognition as sr
from deep_translator import GoogleTranslator
import re

# models


def get_text_input():
    print("Select action:")
    print("0. Quit")
    print("1. Read text from Console")
    print("2. Read text from File")
    print("3. Recognize text from Photo (English only)")
    print("4. Recognize text from Videofile")
    print("5. Recognize text from Audiofile")

    choice = input("Enter action number: ")

    if choice == '0':
        return None

    if choice == '1':
        return input("Enter text: ")
    
    filepath = input("Enter filepath: ")

    if choice == '2':
        encode = input("Enter encoding: ")
        with open(filepath, 'r', encoding=encode) as file:
            return file.read()
    
    if choice == '3':
        return pytesseract.image_to_string(Image.open(filepath))
    
    if choice != '4' and choice != '5':
        print("Incorrect choice. Try again")
        print()
        return get_text_input()
        
    lang = input("Input language code in format like en-EN: ")

    if choice == '4':
        clip = mp.VideoFileClip(filepath)
        filepath += '.wav'
        clip.audio.write_audiofile(filepath, codec='pcm_s16le')

    r = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio = r.record(source)
        text = r.recognize_google(audio, language=lang)
        return text

def main():
    text = None
    try:
        text = get_text_input()
    except Exception as e:
        print(f"Something went wrong: {type(e).__name__}")
    if text == None:
        return -1
    
    print()

    print("Text is positive!")

    print(text)
    return 0

if __name__ == "__main__":

    active = True
