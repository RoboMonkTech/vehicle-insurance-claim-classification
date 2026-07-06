from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
model_package = joblib.load("models/insurance_pipeline.pkl")

model = model_package["model"]
threshold = model_package["threshold"]


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict(input_data):

    df = pd.DataFrame([input_data])

    prob = model.predict_proba(df)[:, 1][0]

    pred = 1 if prob >= threshold else 0

    return pred, prob


# ==========================================================
# HOME ROUTE
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# PREDICTION ROUTE
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict_route():

    try:

        data = {
            "Gender": request.form.get("Gender"),
            "Age": int(request.form.get("Age", 0)),
            "Driving_License": int(request.form.get("Driving_License", 1)),
            "Region_Code": float(request.form.get("Region_Code", 0)),
            "Previously_Insured": int(request.form.get("Previously_Insured", 0)),
            "Vehicle_Age": request.form.get("Vehicle_Age"),
            "Vehicle_Damage": request.form.get("Vehicle_Damage"),
            "Annual_Premium": float(request.form.get("Annual_Premium", 0)),
            "Policy_Sales_Channel": float(request.form.get("Policy_Sales_Channel", 0)),
            "Vintage": int(request.form.get("Vintage", 0))
        }

        pred, prob = predict(data)

        result = "Claim Likely" if pred == 1 else "No Claim Expected"

        confidence = round(prob * 100, 2)

        return render_template(
            "index.html",
            prediction=result,
            probability=confidence
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            probability=0
        )


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)