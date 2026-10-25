"""Feature inventory is intentionally limited to documented origination-time fields."""
from .data_preprocessing import available_features

def feature_documentation(df):
    numeric, categorical = available_features(df)
    return {"numeric": numeric, "categorical": categorical,
            "excluded": ["default", "loan_sequence_number", "first_report", "last_report"],
            "note": "No borrower income, occupation, or home-ownership field is provided in Freddie Mac's public loan-level sample layout."}
