import base64
import cv2
import numpy as np
import pandas as pd
import tempfile
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from fastapi import APIRouter, FastAPI, Form
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

# Load the trained model
model = load_model('pickle\\handwrite_model.h5')
dictionary_path = "datasets\\Dictionary.xlsx"
label_encoder_path = "pickle\\label_encoder.pkl"

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

app = FastAPI()

app.include_router(textIdentify, tags=["Image Prediction"])