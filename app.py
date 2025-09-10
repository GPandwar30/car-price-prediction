from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load pipeline (preprocessing + model)
model = pickle.load(open("car_model.pkl", "rb"))

# Load cleaned dataset for dropdowns
car_data = pd.read_csv("Cleaned_Car_data.csv").drop(columns=['Unnamed: 0'], errors='ignore')

companies = sorted(car_data['company'].unique())
car_models = sorted(car_data['name'].unique())
fuel_types = sorted(car_data['fuel_type'].unique())

@app.route('/')
def home():
    return render_template("index.html",
                           companies=companies,
                           car_models=car_models,
                           fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from form
        name = request.form['name']
        company = request.form['company']
        fuel_type = request.form['fuel_type']
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])

        # Create DataFrame for prediction
        input_df = pd.DataFrame(
            [[name, company, fuel_type, year, kms_driven]],
            columns=['name','company','fuel_type','year','kms_driven']
        )

        # Predict
        prediction = model.predict(input_df)

        return render_template("index.html", 
                               companies=companies,
                               car_models=car_models,
                               fuel_types=fuel_types,
                               prediction_text=f"💰 Estimated Car Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        return render_template("index.html", 
                               companies=companies,
                               car_models=car_models,
                               fuel_types=fuel_types,
                               prediction_text=f"⚠ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
