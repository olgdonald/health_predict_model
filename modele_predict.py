import pandas as pd
import joblib

# Charger le modèle, l'encodeur et les colonnes
model = joblib.load("trained_health_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
expected_columns = joblib.load("columns.joblib")

# Exemple de nouvelles données avec des symptômes 
new_data = pd.DataFrame({
    'itching': [1],
    'skin_rash': [1],
    'nodal_skin_eruptions': [0],
    'high_fever': [1],
    'blister': [1],
    'red_spots_over_body': [1],
    'pus_filled_pimples': [0],
    'sunken_eyes': [0],
    'stomach_pain': [0],
    'headache': [1],
    'loss_of_appetite': [1],
    'yellow_crust_ooze': [0],
    'sore_throat': [0]
    # Ajoutez d'autres colonnes nécessaires...
})


new_data = new_data.reindex(columns=expected_columns, fill_value=0)

# Prédire
prediction = model.predict(new_data)
predicted_class = label_encoder.inverse_transform(prediction)

print("Prédiction :", predicted_class[0])
