import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model as load_saved_model  # Rename load_model function

# Load the saved model (load the model only when needed)
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        return load_saved_model("grand_final_recognition_model.h5")  # Use the renamed function
    except Exception as e:
        st.error("Failed to load the model. Please try again later.")
        st.write(e)
        return None

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Display error message
def display_error_message(message):
    st.error(message)

# Display success message
def display_success_message(message):
    st.success(message)

# Create a Streamlit web application
st.title("Flower Recognition App")

# Add description/instructions for users
st.write("This app uses a pre-trained deep learning model to recognize different types of flowers. "
         "Please upload an image of a flower (in JPG or PNG format), and the app will predict its type.")

# Allow users to upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and make predictions
    model = load_model()
    if model is not None:
        img_array = preprocess_image(image)
        try:
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence_score = prediction[0][predicted_class] * 100
            # Map predicted class index to flower name
            flower_names = ['Daisy', 'Sunflower', 'Tulip', 'Dandelion', 'Rose']
            predicted_flower = flower_names[predicted_class]
            # Display the prediction with confidence score
            st.write("Predicted Flower:", predicted_flower)
            st.write("Confidence Score:", f"{confidence_score:.2f}%")
        except Exception as e:
            display_error_message("An error occurred during prediction. Please try again.")
            st.write(e)
    else:
        display_error_message("Failed to load the model. Please try again later.")
