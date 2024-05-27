from xgboost import XGBClassifier

def train_model(X_train, y_train):
    model = XGBClassifier(colsample_bytree=0.8,
                          learning_rate=0.1,
                          max_depth=7,
                          n_estimators=200,
                          subsample=1.0,
                          random_state=42)
    model.fit(X_train, y_train)
    return model