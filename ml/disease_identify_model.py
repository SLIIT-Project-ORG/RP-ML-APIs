import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import pickle

datasets_path = Path(__file__).resolve().parent.parent / 'datasets'
pickle_files_path = Path(__file__).resolve().parent.parent / 'pickle_files'

knowledge_base = pd.read_excel(datasets_path / 'knowledge_base.xlsx')
new_data = pd.read_excel(datasets_path / 'new_data.xlsx')
disease_model_path = pickle_files_path / 'disease_model.pkl'

knowledge_base = pd.concat([knowledge_base, new_data], ignore_index=True)
# knowledge_base.drop_duplicates(inplace=True)
knowledge_base.dropna(inplace=True)

X = knowledge_base['symptoms']
y = knowledge_base['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using CountVectorizer with the default tokenizer (word-level tokenizer)
vectorizer = CountVectorizer()

model = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

model_dict = {
    'model': model
}

# Save the dictionary containing the model using pickle
with open(disease_model_path, 'wb') as file:
    pickle.dump(model_dict, file)
    print(f"Model saved to {disease_model_path}")

# # Load the saved dictionary using pickle
with open(disease_model_path, 'rb') as file:
    loaded_model_dict = pickle.load(file)

# # # Use the loaded model for predictions
new_symptoms = ["කැස්ස,සෙම්ප්‍රතිශ්‍යාව","පිපාසය,මහන්සිය","හිස,රදය,වමනය","බිබිලි,තුවාල"]
# new_symptoms = ["කැස්ස,සෙම්ප්‍රතිශ්‍යාව"]

predicted_disease = loaded_model_dict['model'].predict(new_symptoms)

print(f"Predicted disease: {predicted_disease[0]}")
print(f"Predicted disease: {predicted_disease[1]}")
print(f"Predicted disease: {predicted_disease[2]}")
print(f"Predicted disease: {predicted_disease[3]}")

