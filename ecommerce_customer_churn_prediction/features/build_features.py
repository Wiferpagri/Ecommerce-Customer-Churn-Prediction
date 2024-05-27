import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from ecommerce_customer_churn_prediction.utils.imbalanced_learn_master.imblearn.over_sampling._smote import SMOTE


def preprocess_data(train_data, test_data):
    
    df = train_data.copy(deep=True)
    test_df = test_data.copy(deep=True)
    # Drop ID column
    df = df.drop('CustomerID', axis=1)
    
    # Drop original discarded columns
    discard_cols = ['NumberOfAddress',
                    'SatisfactionScore',
                    'DaySinceLastOrder',
                    'CashbackAmount',
                    'WarehouseToHome',
                    'NumberOfDeviceRegistered',
                    'OrderAmountHikeFromlastYear',
                    'OrderCount',
                    'CouponUsed',
                    'HourSpendOnApp',
                    'Gender']
    df = df.drop(discard_cols, axis=1)
    test_df = test_df.drop(discard_cols, axis=1)
    
    # Correcting typo errors on train data
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace('Mobile Phone', 'Phone')

    poc_typos = {'Mobile Phone' : 'Phone',
                'Mobile' : 'Phone'}
    df['PreferedOrderCat'] = df['PreferedOrderCat'].replace(poc_typos)

    ppm_typos = {'Credit Card' : 'CC',
                'Cash on Delivery' : 'COD'}
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace(ppm_typos)
    
    # Changing dtypes
    bool_cols = ['Churn', 'Complain']
    df[bool_cols] = df[bool_cols].astype(bool)
    cat_cols = ['MaritalStatus', 'PreferedOrderCat', 'CityTier', 'PreferredPaymentMode', 'PreferredLoginDevice']
    df[cat_cols] = df[cat_cols].astype('category')
    
    test_df[bool_cols] = test_df[bool_cols].astype(bool)
    test_df[cat_cols] = test_df[cat_cols].astype('category')
    
    # Missing values treatment
    df = df.fillna(df.median(numeric_only=True))
    
    # Outliers treatment
    out_cols = ['Tenure']
    
    for col in out_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # One-Hot Encoding
    ohe_encoder = OneHotEncoder(sparse_output=False)
    ohe_encoder_test = OneHotEncoder(sparse_output=False)
    
    X_encoded = pd.DataFrame(ohe_encoder.fit_transform(df[cat_cols]))
    X_encoded.columns = ohe_encoder.get_feature_names_out(cat_cols)
    X = df.drop(cat_cols, axis=1).reset_index(drop=True)
    X = pd.concat([X, X_encoded], axis=1)
    
    X_test_encoded = pd.DataFrame(ohe_encoder_test.fit_transform(df[cat_cols]))
    X_test_encoded.columns = ohe_encoder_test.get_feature_names_out(cat_cols)
    X_test = df.drop(cat_cols, axis=1).reset_index(drop=True)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)
        
    # Training and test data split
    y = X['Churn']
    
    y_test = X_test['Churn']
    
    # Select only selected features for the model
    selected_features = ['Complain',
                        'Tenure',
                        'MaritalStatus_Single',
                        'PreferedOrderCat_Phone',
                        'MaritalStatus_Married',
                        'PreferedOrderCat_Laptop & Accessory',
                        'CityTier_3',
                        'PreferedOrderCat_Fashion',
                        'MaritalStatus_Divorced',
                        'CityTier_1',
                        'PreferredPaymentMode_COD',
                        'PreferredLoginDevice_Computer',
                        'PreferredPaymentMode_CC'
                        ]
        
    X = X[selected_features]

    X_test = X_test[selected_features]
    
    # Applying SMOTE on training dataset
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Scaling variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    
    scaler_test = StandardScaler()
    X_test_scaled = scaler_test.fit_transform(X_test)
    
    return X_scaled, y_resampled, X_test_scaled, y_test