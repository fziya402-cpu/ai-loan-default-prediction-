
from flask import Flask, render_template, request
from model import predict_risk, get_model_evaluation
import pandas as pd

app = Flask(__name__)


# Home page: loan application form
@app.route("/")
def home():
    return render_template("index.html")


# Predict the loan risk
@app.route("/predict", methods=["POST"])
def predict():
    try:
        income = float(request.form["income"])
        credit_score = float(request.form["credit_score"])
        loan_amount = float(request.form["loan_amount"])
        previous_default = request.form["previous_default"]
        loan_term = float(request.form["loan_term"])
        age = float(request.form["age"])

        risk = predict_risk(
            income,
            credit_score,
            loan_amount,
            previous_default,
            loan_term,
            age
        )

        return render_template(
            "result.html",
            risk=risk,
            income=income,
            credit_score=credit_score,
            loan_amount=loan_amount,
            previous_default=previous_default,
            loan_term=loan_term,
            age=age
        )

    except Exception as error:
        return f"Prediction failed: {error}", 500


# Analytics dashboard
@app.route("/analytics")
def analytics():
    data = pd.read_csv("loan_data.csv")

    risk_counts = data["Risk_Level"].value_counts()

    return render_template(
        "analytics.html",
        total=len(data),
        low=int(risk_counts.get("Low", 0)),
        medium=int(risk_counts.get("Medium", 0)),
        high=int(risk_counts.get("High", 0))
    )


# Model evaluation page
@app.route("/evaluation")
def evaluation():
    results = get_model_evaluation()

    return render_template(
        "evaluation.html",
        results=results
    )


# Start the website
if __name__ == "__main__":
    app.run(debug=True)
