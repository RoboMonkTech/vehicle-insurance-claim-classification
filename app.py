from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ==========================================================
# LOAD MODEL
# ==========================================================

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
            "Age": int(request.form.get("Age")),
            "Driving_License": int(request.form.get("Driving_License")),
            "Region_Code": float(request.form.get("Region_Code")),
            "Previously_Insured": int(request.form.get("Previously_Insured")),
            "Vehicle_Age": request.form.get("Vehicle_Age"),
            "Vehicle_Damage": request.form.get("Vehicle_Damage"),
            "Annual_Premium": float(request.form.get("Annual_Premium")),
            "Policy_Sales_Channel": float(request.form.get("Policy_Sales_Channel")),
            "Vintage": int(request.form.get("Vintage"))
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
            prediction="Invalid Input / Error Occurred",
            probability=0
        )

# ==========================================================
# RUN APP (DEPLOYMENT SAFE)
# ==========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=False)