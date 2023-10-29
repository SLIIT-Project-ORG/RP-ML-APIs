import base64
import os
import cv2
import numpy as np
import pandas as pd
import tempfile
import pickle
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from fastapi import APIRouter, FastAPI, Form
from fastapi.responses import JSONResponse
from io import BytesIO
from gtts import gTTS as gtts
from PIL import Image, ImageDraw, ImageFont

# Load the trained model
model = load_model('pickle_files/handwrite_model.h5')
dictionary_path = "datasets/Dictionary.xlsx"
label_encoder_path = "pickle_files/label_encoder.pkl"

# Define the image size
img_size_w = 50
img_size_h = 20

# Load the label encoder
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Check if the loaded object is a LabelEncoder instance
if not isinstance(label_encoder, LabelEncoder):
    raise ValueError("The loaded object is not a LabelEncoder instance.")

# Define a temporary directory to save uploaded files
temp_dir = tempfile.mkdtemp()

textIdentify = APIRouter()

# Route to handle base64-encoded image and make predictions
# http://localhsot:8000/text-identify/predict
@textIdentify.post("/predict")
async def predict_base64_image(file: str = Form(...)):
    try:
        # Decode the base64 image
        image_data = BytesIO(base64.b64decode(file))
        img = Image.open(image_data).convert("L")  # Convert to grayscale

        # Preprocess the image
        img_array = np.array(img)
        new_array = cv2.resize(img_array, (img_size_w, img_size_h))
        array = new_array.reshape(-1, img_size_w, img_size_h, 1)

        # Make prediction using the model
        pred = model.predict(array)

        # Get the predicted class index
        y = np.argmax(pred)

        # Get the corresponding class label from the label encoder
        predicted_class = label_encoder.inverse_transform([y])[0]

        result_list = predicted_class.split(",")
        capital_result_list = [i.upper() for i in result_list]

        # Set dictionary values to predicted result
        df = pd.read_excel(dictionary_path)
        data_dictionary = {row["ID"]: row["LABEL"] for index, row in df.iterrows()}

        result_list = [data_dictionary[x] for x in capital_result_list if x in data_dictionary]

        return JSONResponse(content={"predicted_class": result_list})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# app = FastAPI()

# app.include_router(textIdentify, tags=["Image Prediction"])

#===============================================================================================================

class TextInput(BaseModel):
    text: str

# http://localhost:8000/text-identify/text-to-speech
@textIdentify.post("/text-to-speech")
async def text_to_speech(text_input: TextInput):
    myText = text_input.text
    language = 'si'
    
    audio = gtts(text=myText, lang=language, slow=False)
    audio.save("audio.mp3")

    with open("audio.mp3", "rb") as audio_file:
        audio_data = audio_file.read()

    # Encode the image data in base64
    base64_voice = base64.b64encode(audio_data)
    
    print("Path : ", "audio.mp3")
    return {
        "response": "Audio saved successfully",
        "voice": base64_voice,
        "status": "200"
    }
        
    # audio.save("audio.mp3")
    
    # audio_path = os.path.join(os.getcwd(), "audio.mp3")
    # print("Path : ", audio_path)
    # return {
    #     "response": audio_path,
    #     "status": "success"
    # }
    

#===============================================================================================================
# http://localhsot:8000/text-identify/text-to-image
@textIdentify.post("/text-to-image")
async def text_to_image(text_input: TextInput):
    myText = text_input.text
    
    width, height = 1500, 600

    # Create a new image with white background
    image = Image.new("RGB", (width, height), "white")

    # Load a font (you can change the font path and size)
    font = ImageFont.truetype("other/font/iskpotab.ttf", 36)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define text position and color
    text_position = (50, 50)
    text_color = (0, 0, 0)  # Black color

    # Draw the text on the image
    draw.text(text_position, myText, font=font, fill=text_color)

    # Save the image
    image.save("output_image.png")

    with open("output_image.png", "rb") as image_file:
        image_data = image_file.read()

    # Encode the image data in base64
    base64_image = base64.b64encode(image_data).decode('utf-8')

    print("Image saved as 'output_image.png'")
    
    return {
        "response": "successfully downloaded",
        "status": "200",
        "image": base64_image
    }


@textIdentify.get("/hello")
async def hello():
    return {
        "response": "hello",
        "status": "200"
    }