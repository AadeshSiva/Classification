import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\Aadesh\OneDrive\Desktop\PONGAL\dynamic_pricing.csv")

def categorize_cost(cost):
    if cost < data['Historical_Cost_of_Ride'].quantile(0.33):
        return 0 
    elif cost < data['Historical_Cost_of_Ride'].quantile(0.66):
        return 1  
    else:
        return 2  

data['Price_Category'] = data['Historical_Cost_of_Ride'].apply(categorize_cost)

for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category').cat.codes

X = data.drop(columns=['Historical_Cost_of_Ride', 'Price_Category']).values
y = data['Price_Category'].values

def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - test_size))
    return X[indices[:split_idx]], X[indices[split_idx:]], y[indices[:split_idx]], y[indices[split_idx:]]

X_train, X_test, y_train, y_test = train_test_split(X, y)

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = distances.argsort()[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            predictions.append(np.bincount(k_nearest_labels).argmax())
        return predictions

model = KNNClassifier(k=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
