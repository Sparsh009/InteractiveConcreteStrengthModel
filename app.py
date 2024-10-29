import pandas as pd
import numpy as np
import gradio as gr
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
from PIL import Image

# Load and prepare the dataset
data = pd.read_csv('concrete_data.csv')

# Strip whitespace from column names (if any)
data.columns = data.columns.str.strip()

# Feature names
feature_names = ['cement', 'blast_furnace_slag', 'fly_ash', 'water',
                 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']

# Splitting the data into features and target
X = data[feature_names]
y = data["concrete_compressive_strength"]

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the EBM model
ebm = ExplainableBoostingRegressor(feature_names=feature_names, interactions=0)
ebm.fit(X_train, y_train)

# Define prediction function
def predict_strength(cement, blast_furnace_slag, fly_ash, water, superplasticizer,
                     coarse_aggregate, fine_aggregate, age):
    input_data = pd.DataFrame({
        'cement': [cement],
        'blast_furnace_slag': [blast_furnace_slag],
        'fly_ash': [fly_ash],
        'water': [water],
        'superplasticizer': [superplasticizer],
        'coarse_aggregate': [coarse_aggregate],
        'fine_aggregate': [fine_aggregate],
        'age': [age]
    })
    input_data_scaled = scaler.transform(input_data)
    prediction = ebm.predict(input_data_scaled)
    return prediction[0]

# Create the Gradio interface
gradio_interface = gr.Interface(
    fn=predict_strength,
    inputs=[gr.Number(label=name) for name in feature_names],
    outputs="number",
    title="Concrete Strength Prediction with EBM",
    description="Enter material content to predict the concrete strength using EBM."
)

if __name__ == "__main__":
    gradio_interface.launch()
