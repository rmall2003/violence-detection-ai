import os
import cv2
import numpy as np
import tensorflow as tf
import imageio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

DATASET_DIR = "./dataset"
VIOLENCE_DIR = os.path.join(DATASET_DIR, "Violence")
NON_VIOLENCE_DIR = os.path.join(DATASET_DIR, "NonViolence")
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = 224
FRAMES_PER_VIDEO = 20
BATCH_SIZE = 32
EPOCHS = 10

def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    count = 0
    while len(frames) < num_frames and count < total_frames:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frames.append(frame)
        except Exception as e:
            print(f"Error while extracting frame: {e}")
            continue
        count += 1
    cap.release()
    return np.array(frames)


def load_data():
    X, y = [], []
    print("Loading Violence videos...")
    for video_file in tqdm(os.listdir(VIOLENCE_DIR)):
        if video_file.endswith(".mp4") or video_file.endswith(".avi"):
            frames = extract_frames(os.path.join(VIOLENCE_DIR, video_file))
            if len(frames) == FRAMES_PER_VIDEO:
                X.append(frames)
                y.append(1)

    print("Loading Non-Violence videos...")
    for video_file in tqdm(os.listdir(NON_VIOLENCE_DIR)):
        if video_file.endswith(".mp4") or video_file.endswith(".avi"):
            frames = extract_frames(os.path.join(NON_VIOLENCE_DIR, video_file))
            if len(frames) == FRAMES_PER_VIDEO:
                X.append(frames)
                y.append(0)

    X = np.array(X)
    y = np.array(y)
    print(f"Total samples: {len(y)}")
    return X, y

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    inputs = tf.keras.layers.Input((FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3))  
    x = TimeDistributed(base_model)(inputs) 
    x = TimeDistributed(Flatten())(x)
    x = LSTM(64)(x) 
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

checkpoint = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "model-{epoch:02d}-{val_loss:.2f}.keras"),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]
)

model.save("violence_detection_model.h5")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

print("Evaluating the model...")
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.title("Confusion Matrix")
plt.imshow(conf_matrix, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

loaded_model = tf.keras.models.load_model("violence_detection_model.h5")
print("Model loaded successfully!")

y_pred_loaded = (loaded_model.predict(X_test) > 0.5).astype("int32")
print(f"Accuracy of loaded model: {accuracy_score(y_test, y_pred_loaded):.4f}")
