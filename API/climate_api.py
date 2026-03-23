import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import numpy as np


df = pd.read_csv('climate_change_dataset.csv')
print("Shape of dataset: ", df.shape)
print("missing values: ", df.isnull().values.any())

df.columns = df.columns.str.strip()

print("Columns in dataset:")
print(df.columns)

target = 'Extreme Weather Events'

feature_columns = [
    'Year',
    'Avg Temperature (°C)',
    'CO2 Emissions (Tons/Capita)',
    'Sea Level Rise (mm)',
    'Rainfall (mm)',
    'Population',
    'Renewable Energy (%)',
    'Forest Area (%)',
    'Country'
]

X = df[feature_columns].copy()
y = df[target]

# Encode Country safely
le_country = LabelEncoder()
X['Country'] = le_country.fit_transform(X['Country'])



rf_model = RandomForestClassifier()

rf_model.fit(X,y)




with open("climate_change_dataset.pkl", "wb") as file_obj:
    pickle.dump(rf_model, file_obj)
