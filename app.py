import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = "model/waste_classifier.h5"
if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Please train the model first (run train.py).")
else:
    model = tf.keras.models.load_model(MODEL_PATH)

    class_indices = {0: "Class_0"}
    if os.path.exists("dataset/train"):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        gen = ImageDataGenerator(rescale=1./255)
        flow = gen.flow_from_directory("dataset/train", target_size=(224,224))
        class_indices = {v: k for k, v in flow.class_indices.items()}

    st.title("♻ Waste Classification App")
    st.write("Upload an image to classify it into categories.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = image.load_img(uploaded_file, target_size=(224,224))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class] * 100
        label = class_indices.get(predicted_class, "Unknown")

        st.image(img, caption=f"Prediction: {label} ({confidence:.2f}%)", use_column_width=True)
