import tensorflow as tf
import joblib

# Load the model
model = tf.keras.models.load_model("violence_detection_model.h5")

# Save the model as a .pkl file
joblib.dump(model, "violence_detection_model.pkl")

print("Model saved as .pkl using joblib successfully!")
