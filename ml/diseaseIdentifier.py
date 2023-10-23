from pydantic import BaseModel
import speech_recognition as sr
from textblob import TextBlob
import nltk
from fastapi import APIRouter

# Download nltk
nltk.download('')

identifier = APIRouter()

class VoiceInput(BaseModel):
    voice_input: str

@identifier.post("/voice_input")
async def getVoiceInput(input:VoiceInput):
    
    convertedText = input.voice_input
    record = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        record.adjust_for_ambion_noise(source)
        audio = record.listen(source, timeout=10,pharse_time_limit=5)
    try:
        text = record.recognize_google(audio)
        
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        noun_phrases = blob.noun_phrases
        
        print("Text : " , text)
        
        return{
            "response" : text
        }
        
    except sr.UnknownValueError:
        return{
            "error":"Sorry I could not understand"
        }