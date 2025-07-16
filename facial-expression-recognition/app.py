import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


model = tf.keras.models.load_model("emotion_cnn_model.h5")


emotion_labels = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😄', 'Sad 😢', 'Surprise 😲', 'Neutral 😐']


st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("😊 Facial Emotion Recognition")
st.markdown("Upload a **face image** and get the **predicted emotion** below.")


uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  
    image = ImageOps.fit(image, (48, 48), Image.ANTIALIAS)

    st.image(image, caption="Uploaded Face", use_column_width=False)

    
    img_array = np.asarray(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = np.expand_dims(img_array, -1)      

    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

  
    st.subheader("Prediction:")
    st.write(f"**{emotion_labels[predicted_class]}** ({confidence:.2f}% confidence)")
