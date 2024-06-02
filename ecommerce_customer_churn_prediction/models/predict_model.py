from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    general_accuracy = ((accuracy_score(y_test, y_pred))*100)
    conf_matrix = confusion_matrix(y_test, y_pred)
    pos_accuracy = (conf_matrix[1,1] / conf_matrix[1].sum()).round(5)*100
    neg_accuracy = (conf_matrix[0,0] / conf_matrix[0].sum()).round(5)*100
    
    return general_accuracy, conf_matrix, pos_accuracy, neg_accuracy