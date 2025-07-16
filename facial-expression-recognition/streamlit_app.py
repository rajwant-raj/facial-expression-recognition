import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load trained model
model = tf.keras.models.load_model("emotion_cnn_model.h5")

# Emotion labels (FER-2013)
emotion_labels = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😄', 'Sad 😢', 'Surprise 😲', 'Neutral 😐']

# Page config
st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("😊 Facial Emotion Recognition")
st.markdown("Upload a **face image** and get the **predicted emotion** below:")

# File uploader
uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("L")
        image = ImageOps.fit(image, (48, 48), Image.Resampling.LANCZOS)
        
        st.image(image, caption="Processed Grayscale Face", use_container_width=False)

        img_array = np.asarray(image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
        img_array = np.expand_dims(img_array, -1)      # Channel dimension

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class] * 100

        # Result
        st.subheader("Prediction:")
        st.success(f"**{emotion_labels[predicted_class]}** ({confidence:.2f}% confidence)")
        
        # Optional: Show all class probabilities
        with st.expander("See all emotion probabilities"):
            for idx, prob in enumerate(prediction[0]):
                st.write(f"{emotion_labels[idx]}: {prob*100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error processing the image: {e}")
