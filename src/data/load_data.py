import pandas as pd
import json

def load_data():
    transactions_df = pd.read_csv("data/raw/transactions_data.csv")
    cards_df = pd.read_csv("data/raw/cards_data.csv")
    users_df = pd.read_csv("data/raw/users_data.csv")
    with open("data/raw/train_fraud_labels.json", "r") as f:
        fraud_labels = json.load(f)
    return transactions_df, cards_df, users_df, fraud_labels
