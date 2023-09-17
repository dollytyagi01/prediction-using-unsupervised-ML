import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Load the data from an external CSV file
@st.cache
def load_data():
    # Replace "data.csv" with the path to your external CSV file
    data = pd.read_csv("iris.csv")
    return data

data = load_data()

# Separate features and target variable
X = data.drop(["Id", "Species"], axis=1)
y = data["Species"]
class_names = np.unique(y)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Define the Streamlit app
def main():
    # Set the title and the sidebar
    st.title("Iris Species Classifier")
    st.sidebar.title("Options")

    # Add inputs for Sepal and Petal measurements
    sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X["SepalLengthCm"].min()), float(X["SepalLengthCm"].max()))
    sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X["SepalWidthCm"].min()), float(X["SepalWidthCm"].max()))
    petal_length = st.sidebar.slider("Petal Length (cm)", float(X["PetalLengthCm"].min()), float(X["PetalLengthCm"].max()))
    petal_width = st.sidebar.slider("Petal Width (cm)", float(X["PetalWidthCm"].min()), float(X["PetalWidthCm"].max()))

    # Create a numpy array for the input features
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make predictions using the classifier
    prediction = clf.predict(input_data)

    # Display the predicted class
    st.write(f"Predicted Class: {prediction[0]}")

# Run the Streamlit app
if __name__ == "__main__":
    main()