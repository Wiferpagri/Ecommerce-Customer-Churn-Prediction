import argparse
import warnings
from ecommerce_customer_churn_prediction.data.make_dataset import load_data
from ecommerce_customer_churn_prediction.features.build_features import preprocess_data
from ecommerce_customer_churn_prediction.models.train_model import train_model
from ecommerce_customer_churn_prediction.models.predict_model import evaluate_model

warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument('--train-data-path', type=str, required=True, help='Path to the Excel file containing the data.')
    parser.add_argument('--test-data-path', type=str, required=True, help='Path to the csv file containing the data.')
    return parser.parse_args()

def main():
    args = parse_args()
    train_data = load_data(args.train_data_path, kind='xlsx')
    test_data = load_data(args.test_data_path, kind='csv')
    
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
    model = train_model(X_train, y_train)
    
    print('Model evaluation on the test data:')
    accuracy, conf_matrix, pos_accuracy, neg_accuracy = evaluate_model(model, X_test, y_test)
    print(f'General Accuracy: {accuracy}')
    print(f'Positive Accuracy: {pos_accuracy}')
    print(f'Negative Accuracy: {neg_accuracy}')
    print(f'Confusion Matrix: {conf_matrix}')

if __name__ == "__main__":
    main()