import os

from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import tempfile
import shutil
import pandas as pd

# Load the trained model
model = load_model('pickle\\handwrite_model.h5')
dictionary_path = "datasets\\Dictionary.xlsx"
label_encoder_path = "pickle\\label_encoder.pkl"

# Define the image size
img_size_w = 50
img_size_h = 20

# Load the label encoder
# (Make sure to adjust the path if necessary)
import pickle

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Check if the loaded object is a LabelEncoder instance
if not isinstance(label_encoder, LabelEncoder):
    raise ValueError("The loaded object is not a LabelEncoder instance.")

# Define a temporary directory to save uploaded files
temp_dir = tempfile.mkdtemp()

textIdentify = APIRouter()

# Route to handle image uploads and make predictions
@textIdentify.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded image to the temporary directory
        file_path = f"{temp_dir}/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess the image
        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size_w, img_size_h))
        array = new_array.reshape(-1, img_size_w, img_size_h, 1)

        # Make prediction using the model
        pred = model.predict(array)

        # Get the predicted class index
        y = np.argmax(pred)

        # Get the corresponding class label from the label encoder
        predicted_class = label_encoder.inverse_transform([y])[0]
        print("pred ", predicted_class)
        # Clean up: delete the temporary file
        os.remove(file_path)

        result_list = predicted_class.split(",")
        capital_result_list = []
        for i in result_list:
            capital_result_list.append(i.upper())

        # Set dictionary values to predicted result
        df = pd.read_excel(dictionary_path)

        # print(df.head())
        data_dictionary = {}
        for index, row in df.iterrows():
            id_value = row["ID"]
            label_value = row["LABEL"]
            data_dictionary[id_value] = label_value

        result_list = []
        # print("Dictionary : ", data_dictionary)
        key_list = data_dictionary.keys()
        for x in capital_result_list:
            if key_list.__contains__(x):
                result_list.append(data_dictionary.get(x))

        print("ID List : ", capital_result_list)
        return JSONResponse(content={"predicted_class": result_list})
    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": str(e)})


# Run the FastAPI server using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(textIdentify, host="127.0.0.1", port=8000)
