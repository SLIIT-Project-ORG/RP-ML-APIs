import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

audio_files = []
labels = []

# Dataset folder path
dataset_folder = "D:\\Projects\\SLIIT\\RP\\Datasets\\disease"

# Iterate through disease folders
for disease_folder in os.listdir(dataset_folder):
    print("Folder : ", disease_folder)
    disease_path = os.path.join(dataset_folder, disease_folder)
    # Iterate through audio files in each disease folder
    for audio_file in os.listdir(disease_path):
        audio_path = os.path.join(disease_path, audio_file)
        # Load audio file using librosa
        audio, sr = librosa.load(audio_path)
        audio_files.append(audio)
        labels.append(disease_folder)

mfcc_features = []

# Iterate through audio files
for audio in audio_files:
    # Extract features
    mfcc = librosa.feature.mfcc(y=audio, sr=22040, n_mfcc=13)
    mfcc_features.append(mfcc)

normalized_features = []

# Iterate through features
for mfcc in mfcc_features:
    normalized_mfcc = StandardScaler().fit_transform(mfcc)
    normalized_features.append(normalized_mfcc)

# Pad the features
max_length = max(mfcc.shape[1] for mfcc in normalized_features)
padded_features = []

# Iterate through normalized features
for mfcc in normalized_features:
    # Pad the MFCC feature sequence
    padded_mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    padded_features.append(padded_mfcc)

X = np.array(padded_features)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
print("X train shape : ", X_train.shape)
print("X_test shape  : ", X_test.shape)

# Reshape the training and testing sets
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train the model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predict X test data
y_pred = svm_classifier.predict(X_test)
print("y_pred : ", y_pred)
print("y_test : ", y_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

import pickle

# Save the trained model to a file
model_filename = "D:\\Projects\\SLIIT\\RP\Pickle\\disease_identify_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(svm_classifier, file)