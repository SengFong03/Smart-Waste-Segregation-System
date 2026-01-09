# ♻️ AI-Enhanced Smart Waste Segregation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer%20Vision-purple)
![GCP](https://img.shields.io/badge/Cloud-GCP%20Training-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 Introduction
This project is an **AI-powered Smart Waste Segregation System** developed as a Final Year Project (FYP). It leverages Deep Learning (Computer Vision) to automatically classify waste materials in real-time, aiming to improve recycling efficiency and promote environmental awareness through gamification.

The system utilizes **YOLOv8** for high-accuracy object detection and is deployed via a user-friendly **Streamlit** web interface.

## ✨ Key Features
* **🔍 AI Waste Detection**: Classifies waste into 5+ categories (Plastic, Metal, Paper, Glass, etc.) using a custom-trained YOLOv8 model.
* **📹 Real-Time Webcam Inference**: Supports live video stream detection for immediate feedback.
* **🎮 Eco-Game Module**: A built-in gamified experience where users earn points by correctly sorting waste, tracking high scores in a database.
* **📊 Analytics Dashboard**: Visualizes detection history and waste statistics using SQLite.

## 🛠️ Tech Stack
* **Frontend**: Streamlit
* **Model**: YOLOv8 (Ultralytics), OpenCV
* **Backend & Database**: Python, SQLite
* **Training Infrastructure**: Google Cloud Platform (GCP) Compute Engine (GPU-accelerated)

## 📂 Project Structure
```text
Smart-Waste-Segregation-System/
├── app.py                  # Main application entry point (Streamlit)
├── database.py             # Database handler for waste detection logs
├── game_database.py        # Database handler for the Eco-Game module
├── requirements.txt        # Python dependencies
├── models/
│   └── best.pt          # Custom trained YOLOv8 model weights
├── images/                 # Demo images for testing and screenshots
├── notebooks/              # Jupyter notebooks for analysis (EDA, Cross-Validation)
└── training/               # Python scripts used for GCP training

📸 Screenshots
🔹 Mode 1: Real-Time Webcam Detection
Live inference using the device's camera to detect waste items on the fly.

🔹 Mode 2: Static Image Detection
Upload existing images for classification and analysis.

🚀 Installation & Usage
Clone the repository:

Bash

git clone [https://github.com/SengFong03/Smart-Waste-Segregation-System.git](https://github.com/SengFong03/Smart-Waste-Segregation-System.git)
cd Smart-Waste-Segregation-System
Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Bash

streamlit run app.py
💡 Tip: You can use the sample images provided in the images/ folder to test the classification immediately!

🧠 Methodology (CRISP-DM)
The project development followed the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

Data Understanding: Conducted Exploratory Data Analysis (EDA) to ensure class balance (see notebooks/01_Exploratory_Data_Analysis.ipynb).

Modeling: Trained the YOLOv8 model on Google Cloud Platform (GCP) using NVIDIA GPUs for optimal performance (scripts in training/).

Evaluation: Performed rigorous 5-Fold Cross-Validation to ensure model robustness (see notebooks/03_Cross_Validation.ipynb).

📜 License
Distributed under the MIT License. See LICENSE for more information.

Developed by Cheng Seng Fong