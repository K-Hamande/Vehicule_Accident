from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import joblib
import io

# Initialisation de l'application FastAPI
app = FastAPI()

# Middleware CORS pour permettre l'accès depuis Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des modèles
feature_extractor = tf.keras.models.load_model("Assurance_efficientnet_feature_extractor.h5")
classifier = joblib.load("Assurance_classifier.pkl")

# Noms des labels
label_names = [
    "bonnet-dent",
    "doorouter-dent",
    "fender-dent",
    "front-bumper-dent",
    "rear-bumper-dent"
]

# Route de prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire le fichier image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        # Prétraitement de l'image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extraction des caractéristiques
        features = feature_extractor.predict(img_array)

        # Mise en forme des caractéristiques pour le classificateur
        features_flat = features.reshape((1, -1))

        # Prédiction
        prediction = classifier.predict(features_flat)

        # Conversion en liste d'étiquettes
        predicted_labels = [label_names[i] for i in range(len(label_names)) if prediction[0, i] == 1]

        return {"predictions": predicted_labels}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
