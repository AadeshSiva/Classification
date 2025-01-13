import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\Aadesh\OneDrive\Desktop\AI\dynamic_pricing.csv")

y = data['Historical_Cost_of_Ride'].values

for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category').cat.codes

X = data.drop(columns=['Historical_Cost_of_Ride']).values

X = np.c_[np.ones(X.shape[0]), X] 

def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - test_size))
    return X[indices[:split_idx]], X[indices[split_idx:]], y[indices[:split_idx]], y[indices[split_idx:]]

X_train, X_test, y_train, y_test = train_test_split(X, y)

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X_transpose = X.T
        self.weights = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

    def predict(self, X):
        return X @ self.weights

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse:.2f}")
