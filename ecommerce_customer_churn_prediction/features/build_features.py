import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from category_encoders import TargetEncoder
from ecommerce_customer_churn_prediction.utils.imbalanced_learn_master.imblearn.over_sampling import ADASYN
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_data(train_data, test_data):
    
    df = train_data.copy(deep=True)
    test_df = test_data.copy(deep=True)
    # Drop ID column
    df = df.drop('CustomerID', axis=1)
    
    # Reorganizing data
    test_df = test_df.reindex(columns=df.columns.tolist())
    
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
    cat_cols = ['MaritalStatus', 'PreferedOrderCat', 'CityTier', 'PreferredPaymentMode', 'PreferredLoginDevice', 'Gender']
    int_cols = ['Tenure', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']
    
    df[bool_cols] = df[bool_cols].astype(bool)
    df[cat_cols] = df[cat_cols].astype('category')
    
    test_df[bool_cols] = test_df[bool_cols].astype(bool)
    test_df[cat_cols] = test_df[cat_cols].astype('category')
    test_df[int_cols] = test_df[int_cols].astype(int)
    
    # Missing values treatment
    
    # Encoding columns
    ohe_encoder = OneHotEncoder(sparse_output=False)
    encoded_df = pd.DataFrame(ohe_encoder.fit_transform(df[cat_cols]))
    encoded_df.columns = ohe_encoder.get_feature_names_out(cat_cols)
    df = df.drop(cat_cols, axis=1)
    df = pd.concat([df, encoded_df], axis=1)

    # MICE imputation
    mice_imputer = IterativeImputer(estimator=BayesianRidge(),
    initial_strategy='mean',
    imputation_order='descending',
    random_state=42,
    min_value = 0
    )
    df.iloc[:, :] = mice_imputer.fit_transform(df)
    
    # Inverse Transform
    inv_df = pd.DataFrame(ohe_encoder.inverse_transform(df[encoded_df.columns]), columns=cat_cols)
    df = pd.concat([df.drop(columns=encoded_df.columns), inv_df], axis=1)
    df = df.reindex(columns=test_df.columns.tolist())
    
    # Convert data types again
    df[bool_cols] = df[bool_cols].astype(bool)
    df[cat_cols] = df[cat_cols].astype('category')
    df[int_cols] = df[int_cols].astype(int)
    
    # Outliers treatment
    out_cols = ['Tenure', 'WarehouseToHome', 'NumberOfAddress']
    
    for col in out_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    df = df.reset_index(drop=True)
    
    # Encoding categories for preprocessing
    # Defining function to encode
    def encoding_data(df, ohe_cols, target_cols):
    
        """
    This function preprocesses a given DataFrame by applying specified encoding transformations
    to the feature columns. The function separates the features and the target variable,
    fits the encoders on the data, transforms the features, and returns the transformed features 
    as a new DataFrame for further inspection or analysis.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the feature columns and the target variable `Churn`.
    - ohe_cols (list): A list of column names to be one-hot encoded.
    - target_cols (list): A list of column names to be target encoded.
    
    Returns:
    - X_transformed_df (pd.DataFrame): A DataFrame containing the transformed feature columns.
    
    Example Usage:
    >>> transformed_df = encoding_data(df, ['Gender'], ['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus', 'CityTier'])
    >>> print(transformed_df)
    """
    
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # Apply One-Hot Encoding to 'Gender' column
        onehot_encoder = OneHotEncoder(drop='first')
        X_onehot = onehot_encoder.fit_transform(X[ohe_cols])
        X_onehot_df = pd.DataFrame(X_onehot.toarray(), columns=onehot_encoder.get_feature_names_out(ohe_cols))
        
        # Apply Target Encoding to other categorical columns
        target_encoder = TargetEncoder(cols=target_cols)
        X_target_df = target_encoder.fit_transform(X, y)
        
        # Combine encoded features with remaining features
        X_transformed_df = pd.concat([X_target_df.drop(columns=ohe_cols), X_onehot_df], axis=1)
        X_transformed_df['Complain'] = X_transformed_df['Complain'].astype(bool)
            
        return X_transformed_df
    
    y = df['Churn']
    y_test = test_df['Churn']
    
    ohe_columns = ['Gender']
    target_columns = ['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus', 'CityTier']
    
    X = encoding_data(df, ohe_columns, target_columns)
    X_test = encoding_data(test_df, ohe_columns, target_columns)
    
    ecd_col_names = X.columns.tolist()
    
    # Applying ADASYN on training dataset
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    # Scaling variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_scaled = pd.DataFrame(X_scaled, columns=ecd_col_names)
    
    scaler_test = StandardScaler()
    X_test_scaled = scaler_test.fit_transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=ecd_col_names)
    
    # Creating new features
    X_scaled['Tenure**3'] = X_scaled['Tenure']**3
    X_scaled['CityTier**2*Complain'] = X_scaled['Complain'] * X_scaled['CityTier']**2
    X_scaled['Complain*exp(Gender_Male)'] = X_scaled['Complain'] * np.exp(X_scaled['Gender_Male'])
    
    X_test_scaled['Tenure**3'] = X_test_scaled['Tenure']**3
    X_test_scaled['CityTier**2*Complain'] = X_test_scaled['Complain'] * X_test_scaled['CityTier']**2
    X_test_scaled['Complain*exp(Gender_Male)'] = X_test_scaled['Complain'] * np.exp(X_test_scaled['Gender_Male'])
    
    # Selecting features
    selected_columns = ['Tenure', 'MaritalStatus', 'Complain', 'Tenure**3', 'CityTier**2*Complain', 'Complain*exp(Gender_Male)']
    X_selected = X_scaled[selected_columns]
    X_test_selected = X_test_scaled[selected_columns]
    
    # Scaling data to not have negative numbers
    min_max_scaler = MinMaxScaler()
    
    X_final_scaled = min_max_scaler.fit_transform(X_selected)
    X_test_final_scaled = min_max_scaler.transform(X_test_selected)
    
    return X_final_scaled, y_resampled, X_test_final_scaled, y_test