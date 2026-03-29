import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import streamlit as st

# Load model once and reuse it
model = None

def get_best_model():
    global model
    if model is None:
        model = keras.models.load_model("covid_model.h5", compile=False)
        model.make_predict_function() # Nessary for serving
        print('Model loaded. Start serving...')
    return model

st.subheader('Classify the image')
image_file = st.file_uploader('Choose the Image', ['jpg', 'png'])

if image_file is not None:
    image = Image.open(image_file).convert('RGB')
    st.image(image, caption='Input Image', use_column_width=True)

    # Resize and prepare the image
    image = image.resize((150,150),Image.LANCZOS)
    img_array = np.array(image)

    #Ensure image has 3 channels (RGB)
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    #  Expand dimensions for model input
    img_array = np.expand_dims(img_array, axis=0)
    model = get_best_model()

    # Perform prediction
    classes = model.predict(img_array)
    predicted_class = np.argmax(classes)
    classlabel=['Covid', 'Normal']


    #Display the predicted class
    st.markdown(f'<h3?The image is predicted as class {classlabel[predicted_class]}.</h3>', unsafe_allow_html=True)