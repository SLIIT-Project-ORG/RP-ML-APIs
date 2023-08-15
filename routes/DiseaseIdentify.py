import pickle
from ml.disease_identify_model import X_train
import librosa

import numpy as np
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import normalize

from models.DiseaseIdentify import DiseaseIdentify

with open("D:\\Projects\\SLIIT\\RP\Pickle\\disease_identify_model.pkl", 'rb') as file:
    model = pickle.load(file)
    
class PredictionResult(BaseModel):
    disease: str
    
diseaseIdentify = APIRouter()

@diseaseIdentify.post('/predict')
def predict_disease(audio_file: UploadFile = File(...)):
    # Read audio file
    audio, _ = librosa.load(audio_file.file)
    mfcc = librosa.feature.mfcc(y=audio, sr=_, n_mfcc=13)

    normalized_mfcc = StandardScaler().fit_transform(mfcc)

    max_length = normalized_mfcc.shape[1]
    padded_mfcc = np.pad(normalized_mfcc, ((0, 0), (0, max_length - normalized_mfcc.shape[1])), mode='constant')

    reshaped_mfcc = padded_mfcc.reshape(1, -1)

    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

    prediction = model.predict(X_train_reshaped)

    # Return the predicted disease
    return PredictionResult(disease=prediction[0])