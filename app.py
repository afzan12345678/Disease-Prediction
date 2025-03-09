import pandas as pd
import gradio as gr
import pickle
import numpy as np

# Load dataset (only for extracting symptoms)
df = pd.read_csv("datasets.csv")

# Remove "prognosis" since it's not a feature
if "prognosis" in df.columns:
    df = df.drop(columns=["prognosis"])

# Get the correct symptom list
symptom_list = df.columns.tolist()

# Debugging: Print correct feature count
print(f"✅ Using {len(symptom_list)} symptoms for prediction.")
print(f"Symptoms: {symptom_list}")

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to predict disease
def predict_disease(symptoms):
    input_vector = np.zeros(len(symptom_list))

    for symptom in symptoms.split(","):
        symptom = symptom.strip().lower()
        if symptom in symptom_list:
            input_vector[symptom_list.index(symptom)] = 1

    return f"Predicted Disease: {model.predict([input_vector])[0]}"

# Gradio Interface
iface = gr.Interface(fn=predict_disease, inputs=gr.Textbox(), outputs="text")
print("✅ App is running!")

iface.launch()
