import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def train_model():
    # Load data
    data = pd.read_csv('model/data/house_prices.csv')
    X = data[['Area']]
    y = data['Price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    with open('model/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model()
