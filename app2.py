import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

# Function to make predictions
def make_prediction(model, img, flower_classes, top_k=1):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    top_indices = np.argsort(prediction[0])[-top_k:][::-1]
    top_classes = [flower_classes[i] for i in top_indices]
    top_probs = [prediction[0][i] for i in top_indices]
    return top_classes, top_probs

# Load the saved model
model_names = ['flower_model.h5', 'flower_model_v2.h5']  # Add more model names if needed
selected_model = st.sidebar.selectbox("Select Model", model_names)
model = load_model(selected_model)

# Load flower classes
flower_classes = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Sidebar options
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image_size = st.sidebar.slider("Image Size", 50, 300, 150, 50)
top_k = st.sidebar.slider("Top K Predictions", 1, 5, 1)

st.title('Flower Classification App')

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((image_size, image_size))
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Make prediction
    top_classes, top_probs = make_prediction(model, image, flower_classes, top_k=top_k)

    # Display predictions
    st.subheader("Predictions:")
    for i in range(len(top_classes)):
        st.write(f"{i+1}. {top_classes[i]} - Probability: {round(top_probs[i]*100, 2)}%")

    # Plot prediction probabilities
    fig, ax = plt.subplots()
    ax.barh(top_classes, top_probs)
    ax.set_xlabel('Probability (%)')
    ax.set_ylabel('Flower Class')
    ax.set_title('Top Predicted Classes')
    st.pyplot(fig)
