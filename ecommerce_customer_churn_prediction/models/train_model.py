from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def train_model(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model