
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the sample loan dataset
data = pd.read_csv("loan_data.csv")

# Convert previous default values into numbers
data["Previous_Default"] = data["Previous_Default"].map({
    "No": 0,
    "Yes": 1
})

# Convert risk categories into numbers
encoder = LabelEncoder()
data["Risk_Level"] = encoder.fit_transform(data["Risk_Level"])

# Select input features and target
X = data[
    [
        "Income",
        "Credit_Score",
        "Loan_Amount",
        "Previous_Default",
        "Loan_Term",
        "Age"
    ]
]

y = data["Risk_Level"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Predict risk for a new applicant
def predict_risk(
    income,
    credit_score,
    loan_amount,
    previous_default,
    loan_term,
    age
):
    previous_default = 1 if previous_default == "Yes" else 0

    applicant = pd.DataFrame(
        [[
            income,
            credit_score,
            loan_amount,
            previous_default,
            loan_term,
            age
        ]],
        columns=X.columns
    )

    prediction = model.predict(applicant)

    return encoder.inverse_transform(prediction)[0]


# Calculate model evaluation results
def get_model_evaluation():
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(encoder.classes_)))
    )

    return {
        "accuracy": round(accuracy * 100, 2),
        "report": report,
        "matrix": matrix.tolist(),
        "labels": encoder.classes_.tolist()
    }
