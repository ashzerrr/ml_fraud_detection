import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(transactions_df, cards_df, users_df, train_fraud_labels_df):
    merged_df = pd.merge(transactions_df, train_fraud_labels_df, how='left', left_on='id', right_on='transaction_id')
    labeled_merged_df = merged_df.dropna(subset=['Fraud label']).copy()
    labeled_merged_df['Fraud label'] = labeled_merged_df['Fraud label'].apply(lambda x: 1 if x == 'Yes' else 0)

    label_encoder = LabelEncoder()
    labeled_merged_df['errors_encoded'] = label_encoder.fit_transform(labeled_merged_df['errors'])
    labeled_merged_df['mcc_encoded'] = label_encoder.fit_transform(labeled_merged_df['mcc'])
    labeled_merged_df['zip_encoded'] = label_encoder.fit_transform(labeled_merged_df['zip'])
    labeled_merged_df['merchant_state_encoded'] = label_encoder.fit_transform(labeled_merged_df['merchant_state'])
    labeled_merged_df['merchant_city_encoded'] = label_encoder.fit_transform(labeled_merged_df['merchant_city'])
    labeled_merged_df['merchant_id_encoded'] = label_encoder.fit_transform(labeled_merged_df['merchant_id'])
    labeled_merged_df['use_chip_encoded'] = label_encoder.fit_transform(labeled_merged_df['use_chip'])
    labeled_merged_df['amount'] = labeled_merged_df['amount'].replace(r'[\$,]', '', regex=True).astype(float)
    labeled_merged_df['date'] = pd.to_datetime(labeled_merged_df['date']) 
    labeled_merged_df['year'] = labeled_merged_df['date'].dt.year
    labeled_merged_df['month'] = labeled_merged_df['date'].dt.month
    labeled_merged_df['day'] = labeled_merged_df['date'].dt.day
    labeled_merged_df['hour'] = labeled_merged_df['date'].dt.hour
    labeled_merged_df['minute'] = labeled_merged_df['date'].dt.minute
    labeled_merged_df['second'] = labeled_merged_df['date'].dt.second

    # Drop encoded columns
    labeled_merged_df = labeled_merged_df.drop(['use_chip', 'merchant_id', 'merchant_city', 'merchant_state', 'zip', 'mcc', 'errors'], axis=1)

    # Preprocess users_df
    users_df['yearly_income'] = users_df['yearly_income'].replace(r'[\$,]', '', regex=True).astype('float')
    users_df['per_capita_income'] = users_df['per_capita_income'].replace(r'[\$,]', '', regex=True).astype('float')
    users_df['total_debt'] = users_df['total_debt'].replace(r'[\$,]', '', regex=True).astype('float')
    users_df['gender_encoded'] = label_encoder.fit_transform(users_df['gender'])
    users_df = users_df.drop(columns=['birth_year','birth_month','address','gender'])

    # Preprocess cards_df
    cards_df['card_number_encoded'] = label_encoder.fit_transform(cards_df['card_number'])
    cards_df['card_brand_encoded'] = label_encoder.fit_transform(cards_df['card_brand'])
    cards_df['card_type_encoded'] = label_encoder.fit_transform(cards_df['card_type'])
    cards_df['cvv_encoded'] = label_encoder.fit_transform(cards_df['cvv'])
    cards_df['has_chip_encoded'] = label_encoder.fit_transform(cards_df['has_chip'])
    cards_df['card_on_dark_web_encoded'] = label_encoder.fit_transform(cards_df['card_on_dark_web'])
    cards_df['credit_limit'] = cards_df['credit_limit'].replace(r'[\$,]', '', regex=True).astype('float')
    cards_df['expires_year'] = pd.to_datetime(cards_df['expires'], format='%m/%Y').dt.year
    cards_df['expires_month'] = pd.to_datetime(cards_df['expires'], format='%m/%Y').dt.month
    cards_df['acct_open_year'] = pd.to_datetime(cards_df['acct_open_date'], format='%m/%Y').dt.year
    cards_df['acct_open_month'] = pd.to_datetime(cards_df['acct_open_date'], format='%m/%Y').dt.month
    cards_df = cards_df.drop(columns=['card_on_dark_web','has_chip', 'cvv','card_type','card_number','card_brand','expires','acct_open_date'])

    # Merge user and card data
    labeled_merged_df = labeled_merged_df.merge(users_df, how='left', left_on='client_id', right_on='id')
    labeled_merged_df = labeled_merged_df.merge(cards_df, how='left', left_on='card_id', right_on='id')
    labeled_merged_df = labeled_merged_df.drop(['id_x', 'id_y', 'client_id_x','client_id_y','current_age','id'], axis=1)

    return labeled_merged_df
