from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the saved models
try:
    with open('normalizer.pkl', 'rb') as f:
        normalizer = pickle.load(f)
    logger.info("Normalizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading normalizer.pkl: {e}")
    normalizer = None

try:
    with open('random_forest_model.pkl', 'rb') as f:
        best_rf = joblib.load(f)
    logger.info("Random Forest model loaded successfully")
except Exception as e:
    logger.error(f"Error loading random_forest_model.pkl: {e}")
    best_rf = None

# Ordered list of input features used for training (without ID, target is Stage)
input_columns = [
    'N_Days', 'Status', 'Drug', 'Age', 'Sex',
    'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
    'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
    'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin'
]

# Encoding maps (exact mappings used during training)
encode_maps = {
    'Status': {'C': 0, 'CL': 1, 'D': 2},
    'Drug': {'D-penicillamine': 0, 'Placebo': 1, 'Other': 2},
    'Sex': {'M': 0, 'F': 1},
    'Ascites': {'N': 0, 'Y': 1},
    'Hepatomegaly': {'N': 0, 'Y': 1},
    'Spiders': {'N': 0, 'Y': 1},
    'Edema': {'N': 0, 'Y': 1, 'S': 2}
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Collect input data from the form
            input_data = {}
            for col in input_columns:
                val = request.form.get(col, '').strip()
                if col in encode_maps:
                    input_data[col] = val
                else:
                    try:
                        input_data[col] = float(val) if val else 0.0
                        if col == 'Age':
                            logger.debug(f"Age entered in days: {input_data[col]}")
                    except ValueError:
                        input_data[col] = 0.0
                        logger.warning(f"Invalid numeric input for {col}, defaulting to 0.0")

            logger.debug(f"Raw input data: {input_data}")

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Encode categorical features
            for col in encode_maps:
                input_df[col] = input_df[col].map(encode_maps[col])
                if input_df[col].isnull().any():
                    logger.warning(f"Unrecognized value in {col}, defaulting to 0")
                    input_df[col] = input_df[col].fillna(0)

            # Normalize using the loaded normalizer
            if normalizer is None:
                raise Exception("Normalizer not loaded")
            normalized_df = pd.DataFrame(normalizer.transform(input_df[input_columns]), columns=input_columns)
            logger.debug(f"Normalized data shape: {normalized_df.shape}")

            # Predict using the trained best_rf model
            if best_rf is None:
                raise Exception("Random Forest model not loaded")
            prediction = best_rf.predict(normalized_df)
            logger.debug(f"Prediction: {prediction}")

            return jsonify({
                'prediction': int(prediction[0]),  # Ensure integer output
                'interpretation': "Cirrhosis" if prediction[0] == 1 else "No Cirrhosis (Stage 0)"
            })
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500

    return render_template('index.html', columns=input_columns, encode_maps=encode_maps)

if __name__ == '__main__':
    app.run(debug=True)