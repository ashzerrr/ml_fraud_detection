import pandas as pd
import json

def load_datasets():
    transactions_df = pd.read_csv("transactions_data.csv")
    cards_df = pd.read_csv("cards_data.csv")
    users_df = pd.read_csv("users_data.csv")

    with open("mcc_codes.json", "r") as f:
        mcc_codes_df = json.load(f)
    with open("train_fraud_labels.json", "r") as f:
        train_fraud_labels_df = json.load(f)

    train_fraud_labels_df = pd.DataFrame(list(train_fraud_labels_df["target"].items()), columns=['transaction_id', 'Fraud label'])
    train_fraud_labels_df['transaction_id'] = train_fraud_labels_df['transaction_id'].astype(int)

    return transactions_df, cards_df, users_df, mcc_codes_df, train_fraud_labels_df
