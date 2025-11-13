"""Build, deploy and access a model using scikit-learn"""

import pickle
import pandas as pd # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore

df = pd.read_csv("files/input/house_data.csv", sep=",")

features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

target = df[["price"]]
estimador = LinearRegression()
estimador.fit(features, target)

with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump(estimador, file)


