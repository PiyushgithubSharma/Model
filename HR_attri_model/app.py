from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('model/attrition_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        department = request.form['Department']
        age = int(request.form['Age'])
        total_working_years = int(request.form['TotalWorkingYears'])

        # One-hot encode Department (match training format)
        df = pd.DataFrame([{
            'Department': department,
            'Age': age,
            'TotalWorkingYears': total_working_years
        }])

        df_encoded = pd.get_dummies(df)

        # Align with model's expected columns
        model_columns = model.feature_names_in_
        for col in model_columns:
            if col not in df_encoded:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_columns]

        prediction = model.predict(df_encoded)[0]
        return jsonify({'prediction': 'Yes' if prediction == 1 else 'No'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
