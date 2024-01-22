# common
import numpy as np
from PIL import Image
import pytesseract
import moviepy.editor as mp
import speech_recognition as sr
from deep_translator import GoogleTranslator
import re

# models
# common
import numpy as np
from PIL import Image
import pytesseract
import moviepy.editor as mp
import speech_recognition as sr
from deep_translator import GoogleTranslator
import re

# models
import entities as ent
import sentiment as sent


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
    
def extract_entities(words, tags):
    valid_tags = ['geo', 'org', 'per', 'gpe', 'art', 'eve', 'nat']

    entities = {}
    current_entity = []

    for word, tag in zip(words, tags):
        if any(substring in tag.lower() for substring in valid_tags):
            current_entity.append(word)
        else:
            if current_entity:
                entity_str = ' '.join(current_entity)
                entities[entity_str] = entities.get(entity_str, 0) + 1
                current_entity = []

    sorted_entities = sorted(entities.keys(), key=lambda x: entities[x], reverse=True)

    return sorted_entities

def main():
    text = None
    try:
        text = get_text_input()
    except Exception as e:
        print(f"Something went wrong: {type(e).__name__}")
    if text == None:
        return -1
    
    print()

    text_en = GoogleTranslator(source='auto', target='en').translate(text)

    if sent.is_negative(text_en):
        print("Text recognized as negative")

        words = re.sub(r"[^\w\s']", "", text).split()
        tags = ent.get_entities(words)

        if len(tags) != 0:
            print("The following entities are also mentioned in the text:")
            print(extract_entities(words, tags))

        sure = input("Are you sure you want to read this text (y/N)? ")

        if sure[0] == 'N' or sure[0] == 'n':
            print("We are glad that you decided not to take risks!")
            return 0
    else:
        print("Text is positive!")

    print(text)
    return 0

if __name__ == "__main__":
    ent.initialize()
    sent.initialize()

    active = True
    while active:
        print()
        if main() == -1:
            active = False


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
