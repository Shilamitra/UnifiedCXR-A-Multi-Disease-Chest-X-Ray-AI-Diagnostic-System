UnifiedCXR – Multi-Disease Chest X-Ray AI Diagnostic System

UnifiedCXR is a deep learning–based medical imaging system for automated diagnosis of Tuberculosis, Pneumonia, and Cardiomegaly from chest X-ray images. The project uses CNN architectures such as VGG19, VGG16, and EfficientNet-B0 with transfer learning and ensemble techniques to generate reliable diagnostic predictions.

This project was developed as part of the Data Analytics (CS61061) course at IIT Kharagpur.

📂 Project Folder Structure (Required)

The project must follow the exact folder structure below for proper execution:


medical-scan-project/

backend/ mediscan_env/
  
  models/
    efficientnet_cardiomegaly.h5
    tb_detector.h5
    vgg_unfrozen.h5
    
  app.py
  utils.py
  test_model.py
  requirements.txt


frontend/
  index.html
  script.js
  styles.css

test_dataset/
run.txt
setup.bat
.vscode/



⚠️ Important:

The frontend and backend folders must remain separate and at the same level.

The models folder must be inside the backend directory.

Model .h5 files must be placed inside backend/models/.

⚙️ Prerequisites

Python 3.8 – 3.10 (recommended)

pip

Virtual environment support

Windows / Linux / macOS

🚀 Backend Setup & Execution
1️⃣ Navigate to Backend Folder
cd backend

2️⃣ Create Virtual Environment
python -m venv mediscan_env

3️⃣ Activate Virtual Environment

Windows

mediscan_env\Scripts\activate


Linux / macOS

source mediscan_env/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Run the Flask Server
python app.py


The backend will start on:

http://127.0.0.1:5000/

🌐 Frontend Usage

Navigate to the frontend folder

Open index.html in a browser

Upload a chest X-ray image

The frontend sends the image to the Flask backend for prediction

Diagnostic results are displayed on the UI

🧠 Models Used

VGG19 – Tuberculosis classification

VGG16 – Feature extraction for ensemble learning

EfficientNet-B0 – Cardiomegaly detection

Transfer Learning – Reduced training time and improved sensitivity

Ensemble Strategy – Improved robustness and reliability

📊 Dataset

Combined dataset of 16,000+ chest X-ray images

Multiple disease classes

Optimized preprocessing and augmentation pipelines

📌 Notes & Best Practices

Do not upload:

mediscan_env/

Large datasets

Temporary files

Use .gitignore to exclude virtual environments and cache files

Ensure model filenames match those referenced in app.py

🏁 Outcome

UnifiedCXR demonstrates the effectiveness of deep learning, transfer learning, and ensemble models for multi-disease chest X-ray diagnosis, providing a scalable AI-assisted clinical decision support system.
