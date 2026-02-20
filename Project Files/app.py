import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open('rainfall.pkl', 'rb'))
scaler = pickle.load(open('scale.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data as dictionary
        form_data = request.form.to_dict()

        # IMPORTANT:
        # Use ONLY the 13 features used during training
        required_features = [
            float(form_data['MinTemp']),
            float(form_data['MaxTemp']),
            float(form_data['Rainfall']),
            float(form_data['WindGustSpeed']),
            float(form_data['WindSpeed9am']),
            float(form_data['WindSpeed3pm']),
            float(form_data['Humidity9am']),
            float(form_data['Humidity3pm']),
            float(form_data['Pressure9am']),
            float(form_data['Pressure3pm']),
            float(form_data['Temp9am']),
            float(form_data['Temp3pm']),
            float(form_data['RainToday'])
        ]

        # Convert to numpy array
        final_features = np.array(required_features).reshape(1, -1)

        # Apply scaling
        final_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(final_features)

        # Return result page
        if prediction[0] == 1:
            return render_template('chance.html')
        else:
            return render_template('noChance.html')

    except Exception as e:
        return f"Error occurred: {str(e)}"


# Run application
if __name__ == "__main__":
    app.run(debug=True)
