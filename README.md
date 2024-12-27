# Violence Detection AI
This repository contains the codebase for a deep learning-based **Violence Detection System**. The system processes video files to determine if they contain violent or non-violent content, utilizing advanced computer vision and temporal sequence modeling techniques.

## Features:
- Real-Time Video Analysis: Detects violent and non-violent scenes in uploaded videos.
- Deep Learning Pipeline: Combines ResNet50 for spatial feature extraction and LSTM for temporal analysis.
- Web Interface: User-friendly web interface built with Flask and Tailwind CSS.
- Customizable Training: Easily retrain the model with your dataset.
- Pretrained Model Support: Pretrained weights included for immediate use.

## Dataset:
This project uses the [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset), which categorizes videos into two classes:
**Violence** and **Non-Violence**
Download the dataset and organize it in the following structure:
![1](https://github.com/user-attachments/assets/fa8997d8-9666-463f-a650-7a747e8a9fcb)

Place the dataset folder in the root directory of this repository.

## Project Structure:
![2](https://github.com/user-attachments/assets/d71fcabe-27c8-40b3-be25-24450be3a43e)

## Requirements:
To set up and run the project, ensure you have **Python 3.8 or higher** installed.
### Installing Dependencies
1. Install the required Python libraries:
   pip install -r requirements.txt
2. If you prefer manual installation, ensure the following main libraries are installed:
   - TensorFlow/Keras: For deep learning model building.
   - OpenCV: For video processing.
   - Flask: For the web application.
   - Matplotlib: For data visualization.
   - Scikit-learn: For preprocessing and evaluation metrics.
   **Install manually using:**
     pip install tensorflow opencv-python flask matplotlib scikit-learn flask-cors

## How to use:
1. **Clone the Repository-**
   - git clone https://github.com/rmall2003/violence-detection-ai.git
   - cd violence-detection-ai
3. **Download the Dataset-**
   - Download the [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) and place it in the dataset/ folder as described above.
4. **Train the Model-**
   - To retrain the model on your dataset: *python model_training.py*
     - If using VSCode and facing problem with model training, you are suggested to use Juypter Notebook.
5. **Run the Web Application-**
   - Start the Flask application: *python app.py*
6. **Upload Videos-**
   - Use the interface to upload a video for analysis. The AI system will classify it as either **Violence** or **Non-Violence**.

## Technologies Used:
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Backend**: Flask
- **Frontend**: HTML, Tailwind CSS
- **Visualization**: Matplotlib
- **Data Preprocessing**: Scikit-learn
## Screenshots:
# Landing Page-

# Upload Video- 
# Result Prediction-

# Alert-
![Mail Alert](https://github.com/user-attachments/assets/daaea649-e897-42e9-8e9b-d47124983631)

## Contributors:
- **Rashi Mall**: Assisted in data collection from Kaggle and contributed to front-end development, including preprocessing tasks like frame extraction and resizing.
- **Abhinav Mishra**: Designed the UI for video uploads, implemented result displays, and assisted with preprocessing tasks.
- **Tanu Sharma**: Trained and fine-tuned the ResNet model for violence detection and developed the Flask backend to seamlessly connect the model.

If you wish to contribute in this, most welcome
- Fork the repository.
- Create a new branch (git checkout -b feature/your-feature).
- Commit your changes (git commit -m 'Add new feature').
- Push to the branch (git push origin feature/your-feature).
- Open a pull request.


Feel free to customize this further as needed!
