from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Permet les requêtes depuis Flutter

# Charger le modèle et les dépendances
model = joblib.load("trained_health_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
expected_columns = joblib.load("columns.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données de la requête
    input_data = request.json
    
    # Convertir les données en DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)  # Assurez-vous de bien aligner les colonnes

    # Faire une prédiction
    prediction = model.predict(input_df)
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    # Retourner la prédiction
    return jsonify({'disease': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
