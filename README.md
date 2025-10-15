# 🕷️ Tick Classifier — Automatic Classification of *Hyalomma* Ticks

## 📘 Project Description

This project aims to **develop an automatic system for classifying ticks of the *Hyalomma* genus** from digital images.
The main objective is to **predict both the species and sex** of each tick — two key parameters in **entomological** and **epidemiological** studies.

---

## 🧠 Objectives and Contributions

This work follows an integrated scientific and technical approach, bringing several key contributions:

* 🧬 **Implementation of deep convolutional neural networks (ResNet18 and ResNet50)** using **transfer learning**, achieving high performance even with a limited number of images.
* ⚙️ **Multitask architecture with dual outputs**, allowing **simultaneous prediction** of both species and sex.
* 🧩 **Data augmentation** techniques to address dataset imbalance and improve model robustness.
* 🔁 **End-to-end pipeline** covering data preprocessing, model training, evaluation, and prediction.
* 🌐 **Interactive web application (Flask)** connected to a **MongoDB** database for storing and viewing predictions.
* 📊 **Analytical dashboard** for visualizing model performance and interpreting results.

---

## 🧪 Experimental Results

The experiments demonstrated strong performance:

* **Species classification accuracy:** 96%
* **Sex classification accuracy:** 98%

These results confirm the **feasibility and effectiveness** of the proposed approach in an operational context.

---

## 🧰 Technologies Used

| Technology                  | Purpose                                              |
| --------------------------- | ---------------------------------------------------- |
| 🧠 **ResNet18 / ResNet50**  | CNN architectures with transfer learning             |
| 🌐 **Flask**                | Lightweight Python framework for the web application |
| 🗄️ **MongoDB**             | NoSQL database for storing predictions and metadata  |
| 🐍 **Python**               | Core language for pipeline and backend development   |
| 📈 **Matplotlib / Seaborn** | Data visualization and performance analysis          |

---

## 🚀 Key Features

* Upload tick images for automatic classification.
* Simultaneous **species and sex prediction**.
* Results stored and tracked via **MongoDB**.
* Simple and intuitive **web interface**.
* Analytical **dashboard** for model performance monitoring.

---

## 🧭 Project Structure

```
tick_app/
│
├── app.py                 # Main entry point — launches the Flask web application
│
├── model/                 # Contains model files and definitions
│   ├── best_model3.pth    # Trained ResNet model (tracked via Git LFS)
│   └── resnet_model.py    # Model architecture and loading logic
│
├── static/                # Frontend assets (CSS, JS, images)
│
├── templates/             # HTML templates rendered with Jinja2
│
├── routes/                # Flask route definitions (API endpoints, UI routes, etc.)
│
├── utils/                 # Utility and helper functions (data preprocessing, prediction, etc.)
│
├── services/              # Business logic or service layer (e.g., database handling, analytics)
│
├── ven/                   # Virtual environment (usually not included in Git)
│
├── README.md              # Project documentation
│
└── requirements.txt       # List of Python dependencies

```

---

## 📦 Installation

```bash
git clone https://github.com/<your_username>/tick-classifier.git
cd tick-classifier
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
python app.py
```

Then open your browser and go to **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

