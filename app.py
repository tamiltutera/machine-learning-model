import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the pre-trained model
model = joblib.load('logistic_regression_model.joblib')

# Initialize the Flask application
app = Flask(__name__)

@app.route('/diabetics_predict', methods=['POST'])
def predict():
    print("Prediction request received.")
    try:
        data = request.get_json(force=True)
        print(f"Received data: {data}")

        # Define the expected feature order based on the training data
        feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Convert the incoming JSON data to a Pandas DataFrame, ensuring correct column order
        input_df = pd.DataFrame([data], columns=feature_order)

        # Make prediction using the loaded model
        prediction = model.predict(input_df)

        # Convert prediction to a standard Python int
        prediction_result = int(prediction[0])

        return jsonify({'prediction': prediction_result})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# Run the Flask application
# When running in Colab, you might need to use a tool like ngrok to expose the local server
# For local testing, you can typically run `app.run(debug=True)`

# Instructions for local testing:
# 1. Save the above code as a Python file (e.g., `app.py`).
# 2. Ensure 'logistic_regression_model.joblib' is in the same directory.
# 3. Run the application from your terminal: `python app.py`
# 4. In a new terminal, send a POST request using `curl` or `requests` library.
#    Example using curl:
#    curl -X POST -H "Content-Type: application/json" -d '{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}' http://127.0.0.1:5000/diabetics_predict

# To run within Colab directly (won't expose to external requests without ngrok):
# app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

print("Complete Flask API code generated. Please run this cell to start the API.")

if __name__ == "__main__":
  print("Starting prediction API for diabetes using Flask")
  app.run(debug=True)
