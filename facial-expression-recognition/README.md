# 😊 Facial Emotion Recognition using CNN & Streamlit

This project is a deep learning-based **Facial Emotion Recognition** system that classifies human facial expressions into one of seven categories using the **FER-2013 dataset**. It uses a **Convolutional Neural Network (CNN)** for classification and is deployed as an interactive **Streamlit web app**.

---

## 🚀 Features

- 🧠 Built using **TensorFlow/Keras**
- 📸 Accepts face images in JPG, JPEG, or PNG format
- 🎭 Classifies into 7 emotions:
  - Angry 😠
  - Disgust 🤢
  - Fear 😨
  - Happy 😄
  - Sad 😢
  - Surprise 😲
  - Neutral 😐
- 📊 Displays prediction confidence and class probabilities
- 💻 Lightweight and easy-to-use Streamlit interface

---


## 🛠 Tech Stack

| Category          | Technology             |
|-------------------|------------------------|
| Framework         | Streamlit              |
| Deep Learning     | TensorFlow / Keras     |
| Image Processing  | Pillow (PIL)           |
| Language          | Python                 |
| Dataset           | FER-2013               |

---

## 📁 Folder Structure

facial-emotion-recognition/
│
├── emotion_cnn_model.h5 # Trained CNN model
├── app.py # Streamlit UI code
├── requirements.txt # Python dependencies
├── interface_image.png # App screenshot (optional)
└── README.md # Project documentation


---

## ⚙️ Installation & Run Locally

### 1. Clone the repository
bash
git clone https://github.com/yourusername/facial-emotion-recognition.git
cd facial-emotion-recognition

### 2. Install dependencies
bash

pip install -r requirements.txt

### 3. Run the Streamlit app
bash
streamlit run app.py


---


🧠 Model Training (Optional)
The current model (emotion_cnn_model.h5) is trained on the FER-2013 dataset. If you’d like to retrain:

python

 Load dataset

 Build your CNN model

 Compile and train

 Save model using model.save('emotion_cnn_model.h5')


---


🙋‍♂️ Author
Innocent Lier
🎓 Deep Learning Enthusiast | Student Developer
📫 LinkedIn | GitHub


---
