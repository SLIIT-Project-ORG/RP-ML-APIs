import base64
import pickle
import pandas as pd
from fastapi import APIRouter, UploadFile
from pydantic import BaseModel
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play

diseaseIdentify = APIRouter()

disease_model_path = (
    Path(__file__).resolve().parent.parent / "pickle_files" / "disease_model.pkl"
)
new_data_excel_file_path = "datasets//new_data.xlsx"
knowledge_base_excel_file_path = "datasets//knowledge_base.xlsx"

# =======================================================================================================================

with open(disease_model_path, "rb") as file:
    loaded_model_dict = pickle.load(file)
    loaded_model = loaded_model_dict["model"]

# =======================================================================================================================

doctor_dic = {
    "ආමාශ ආන්ත්රයික": "ආමාශ ආන්ත්රයික විශේෂඥ",
    "දිලීර ආසාදනය": "දිලීර රෝග විශේෂඥ",
    "ඖෂධ ප්රතික්රියාව": "ආමාශ ආන්ත්රයික විශේෂඥ",
    "අසාත්මිකතා": "අසාත්මිකතා විශේෂඥ",
    "දියවැඩියාව": "දියවැඩියා විශේෂඥ",
    "පීනස": "කායික රෝග විශේෂඥ",
}

# =======================================================================================================================


def print_message(disease, doctor):
    return (
        "ඔයාට "
        + disease
        + " රෝගයට අදාළ රෝග ලක්ෂණ කිහිපයක් තිබෙනවා. ඔයාට "
        + doctor
        + " වෛද්‍යවරයෙක් හමුවෙන්න පුලුවන්නම් මේ පිලිබදව වැඩි විස්තර දැනගන්න පුළුවන්. "
    )


# =======================================================================================================================


class SymptomsList(BaseModel):
    symptoms: str


@diseaseIdentify.post("/predict")
async def predict_disease(symptoms_list: SymptomsList):
    symptoms = symptoms_list.symptoms.split(",")
    print("Seperated list : ", symptoms)

    if len(symptoms) <= 1:
        return {
            "response": "නිර්දේශයක් ලබා දීමට ඔබ ලබාදී ඇති විස්තර ප්‍රමාණවත් නොවේ.",
            "status": "400",
        }
    else:
        # Predict using the loaded model
        predicted_disease = loaded_model.predict(symptoms)
        disease = predicted_disease[0]
        print("Disease : ", disease)
        doc = ""
        for i in doctor_dic.keys():
            if disease == i:
                doc = doctor_dic[i]
                print("Doctor : ", doc)

        return {"response": print_message(disease, doc), "status": "200"}


# ======================================================================================================================


class UserInput(BaseModel):
    disease: str
    symptom: str


@diseaseIdentify.post("/update-knowledge-base/")
async def update_knowledge_base(user_input: UserInput):
    disease = user_input.disease
    symptom = user_input.symptom

    user_data = {"disease": [disease], "symptoms": [symptom]}

    # Convert the dictionary to a DataFrame
    user_data_df = pd.DataFrame(user_data)

    try:
        existing_data = pd.read_excel(new_data_excel_file_path)
        # Append the new user data to the existing data
        updated_data = pd.concat([existing_data, user_data_df], ignore_index=True)
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        updated_data = user_data_df

    # Save the updated data to the Excel file
    updated_data.to_excel(new_data_excel_file_path, index=False)

    print("User input data saved to Excel.")
    return {"status": "200", "response": "ඔබගේ ඇතුලත් කිරීම සාර්ථකව අවසන් විය."}


# ======================================================================================================================


class VoiceInput(BaseModel):
    voice_input: str


@diseaseIdentify.post("/voice-to-text")
async def voice_input(input: VoiceInput):
    text = input.voice_input

    src = "audio.wav"

    bytearr = base64.b64decode(text)
    with open(src, "wb") as file:
        file.write(bytearr)

    # all the symptoms
    knowledge_base = pd.read_excel(knowledge_base_excel_file_path)
    all_symptoms = knowledge_base["symptoms"].tolist()
    all_symptoms_str = ",".join(all_symptoms)
    symptoms = all_symptoms_str.split(",")
    print("Symptoms : ", symptoms)

    r = sr.Recognizer()

    AudioSegment.from_file(src).export("audio1.wav", format="wav")

    with sr.AudioFile("audio1.wav") as source:
        r.adjust_for_ambient_noise(source)
        print("Converting Audio File to Text...")

        audio = r.listen(source)

    try:
        res_text = r.recognize_google(audio, language="si-LK")

        print("You said: " + res_text)

        # split the text into words
        word_list = res_text.split()
        print("Seperated list : ", word_list)

        # find the symptoms from the word list
        symptoms_list = []
        for i in word_list:
            if i in symptoms:
                symptoms_list.append(i)

        print("Symptoms list : ", symptoms_list)

        if len(symptoms_list) <= 1:
            return {
                "response": "කණගාටුයි, මට තේරුම් ගැනීමට නොහැකි විය.",
                "status": "400",
            }
        else:
            symptoms = ",".join(symptoms_list)
            # Predict using the loaded model
            return {"response": symptoms, "status": "200"}

    except sr.UnknownValueError:
        return {"response": "කණගාටුයි, මට තේරුම් ගැනීමට නොහැකි විය.", "status": "400"}


# ======================================================================================================================