import pandas as pd
import json
import gdown
import os

def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

def load_datasets():
    # File paths and Drive IDs
    transactions_path = "data/raw/transactions_data.csv"
    labels_path = "data/raw/train_fraud_labels.json"

    download_from_drive("1LeFymy8y_JbjMpxYCk_PBlcGQqW8buW2", transactions_path)
    download_from_drive("1RoY7Q7a99tcWLHEFrn-cnmIImrJmiOh8", labels_path)

    # Load data
    transactions_df = pd.read_csv(transactions_path)

    with open(labels_path, "r") as f:
        train_fraud_labels_df = json.load(f)
    
    train_fraud_labels_df = pd.DataFrame(
        list(train_fraud_labels_df["target"].items()),
        columns=["transaction_id", "Fraud label"]
    )
    train_fraud_labels_df['transaction_id'] = train_fraud_labels_df['transaction_id'].astype(int)

    # Assume the others are still local
    cards_df = pd.read_csv("data/raw/cards_data.csv")
    users_df = pd.read_csv("data/raw/users_data.csv")

    return transactions_df, cards_df, users_df, None, train_fraud_labels_df
