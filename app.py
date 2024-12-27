from flask import Flask, request, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
import cv2
import tempfile
import os
import smtplib


app = Flask(__name__)
CORS(app)  

IMG_SIZE = 224
FRAMES_PER_VIDEO = 20 


def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)  
    count = 0
    while len(frames) < num_frames and count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))  
            frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def evaluate_video(video_path):
    model = load_model('violence_detection_model1.h5') 
    print("Model loaded successfully!")

    frames = extract_frames(video_path)
    if len(frames) != FRAMES_PER_VIDEO:
        print(f"Video does not contain the required number of frames ({FRAMES_PER_VIDEO}).")
        return "Error: Incorrect number of frames"
    
    frames = np.expand_dims(frames, axis=0) 
    prediction = model.predict(frames)  
    
    print("Raw Prediction Output:", prediction)
    
    threshold = 0.5  
    if prediction[0] > threshold:
        return "Violence detected"
    else:
        return "Non-Violence detected"



def preprocess_video(file_path, num_frames=20, img_size=(224, 224)):
    cap = cv2.VideoCapture(file_path)
    frames = []
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, img_size)
        frame = frame / 255.0  
        frames.append(frame)
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(np.zeros((img_size[0], img_size[1], 3)))
    
    processed_frames = np.array(frames)
    processed_frames = np.expand_dims(processed_frames, axis=0)  
    
    return processed_frames
@app.route('/about')
def about():
    return render_template("index.html#about")

@app.route('/contactus')
def contactus():
    return render_template("contactus.html")

@app.route('/techno_used')
def techno_used():
    return render_template("techno_used.html")

@app.route('/terms_privacy')
def terms_privacy():
    return render_template("terms_privacy.html")
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    video_file = request.files['videoFile']
    location = request.form['location']  
    video_path = f"./uploads/{video_file.filename}"
    
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    video_file.save(video_path)
    global predic
    prediction = evaluate_video(video_path)
    predic=prediction
    if prediction=="Violence detected":
        notify_police(location)
    os.remove(video_path)
    
    return render_template('index.html', pred=prediction, location=location)

def notify_police(location_details):
    sender_email = "xyz@gmail.com"
    sender_password = "abc"  
    receiver_email = "abc@gmail.com"
   
    message = f"""\
    Subject: Emergency Alert - Violence Detected!

    Violence has been detected at the following location:
    {location_details}

    Please take immediate action."""

    server= smtplib.SMTP("smtp.gmail.com", 587) 
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, message)
    print("Police notified successfully!")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)