import pandas as pd

def engineer_features(df):
    # Flag high amount
    threshold = df['amount'].quantile(0.95)
    df['high_amount'] = (df['amount'] > threshold).astype(int)

    # Time since last transaction
    df = df.sort_values(by='date')
    df['time_since_last_transaction'] = df['date'].diff().dt.total_seconds().fillna(0)

    # Card activity
    card_activity = df.groupby('card_id').agg({
        'amount': ['mean', 'std', 'max'],
        'Fraud label': 'sum'
    })
    card_activity.columns = ['mean_transaction', 'std_transaction', 'max_transaction', 'fraud_count']
    df = df.merge(card_activity, on="card_id", how="left")

    # Merchant fraud rate
    merchant_fraud_rate = df.groupby('merchant_id_encoded')['Fraud label'].mean().fillna(0)
    df['merchant_fraud_rate'] = df['merchant_id_encoded'].map(merchant_fraud_rate)

    return df
