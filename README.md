# 🚨 Factory Guard AI

### AI-Powered Industrial Safety Risk Prediction System

> A production-oriented Machine Learning system designed to proactively identify unsafe industrial conditions and reduce workplace accidents through intelligent risk prediction.

---

## 📌 Problem Statement

Industrial workplaces face significant safety challenges due to:

* Lack of real-time monitoring
* Human error and unsafe practices
* Delayed identification of hazardous conditions

Traditional systems are **reactive**, responding only after incidents occur.

### 🎯 Objective

To build a **proactive AI-driven safety system** that predicts potential risks and enables early intervention.

---

## 🧠 System Overview

Factory Guard AI is an end-to-end Machine Learning application that:

* Ingests industrial condition parameters
* Applies a trained **LightGBM model**
* Predicts risk levels (**Safe / Unsafe**)
* Displays results through an interactive **Streamlit dashboard**

### 🔁 High-Level Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training (LightGBM)
5. Model Evaluation
6. Deployment via Streamlit UI

---

## 🏗️ System Architecture

```text
[Input Data] → [Preprocessing] → [LightGBM Model] → [Prediction] → [Streamlit UI]
```

---

## ⚙️ Tech Stack

| Layer             | Technology          |
| ----------------- | ------------------- |
| Language          | Python              |
| ML Framework      | LightGBM            |
| Data Processing   | Pandas, NumPy       |
| Visualization     | Matplotlib, Seaborn |
| Web Interface     | Streamlit           |
| Model Persistence | Pickle              |
| Version Control   | Git & GitHub        |

---

## 📊 Model Details

### 🔍 Algorithm: LightGBM

A gradient boosting framework optimized for speed and performance.

### ✅ Why LightGBM?

* Handles large-scale structured data efficiently
* Faster training compared to XGBoost
* High accuracy with lower memory usage

### 📈 Performance Metrics

| Metric   | Score     |
| -------- | --------- |
| Accuracy | **~98%**  |
| F1 Score | **~0.87** |
| ROC-AUC  | **~0.98** |

### 🧪 Training Pipeline

* Data Cleaning
* Feature Scaling
* Train-Test Split
* Model Training
* Hyperparameter Tuning
* Evaluation

---

## 📸 Screenshots

### 🔹 Application Interface

![App](screenshots/home.png)

### 🔹 User Input & Prediction

![Prediction](screenshots/prediction.png)

### 🔹 Output Result

![Output](screenshots/output.png)

### 🔹 Performance Visualization

![Results](screenshots/results.png)

---

## 🚀 Getting Started

### 🔧 Prerequisites

* Python 3.8+
* pip

---

### 📥 Installation

```bash
git clone https://github.com/your-username/FACTORY-GUARD-AI.git
cd FACTORY-GUARD-AI
pip install -r requirements.txt
```

---

### ▶️ Run the Application

```bash
streamlit run app.py
```

---

### 🌐 Access

Open in browser:

```
http://localhost:8501
```

---

## 📁 Project Structure

```text
FACTORY-GUARD-AI/
│
├── app.py                # Streamlit application
├── train_model.py       # Model training script
├── requirements.txt     # Dependencies
├── README.md            # Documentation
├── screenshots/         # Project images
└── model.pkl            # Trained ML model
```

---

## 📈 Results & Impact

* Achieves **high prediction accuracy (~98%)**
* Enables **early detection of unsafe conditions**
* Reduces dependency on manual monitoring
* Can significantly improve **industrial safety compliance**

---

## 🔮 Future Scope

* 🎥 Integration with **Computer Vision (CCTV monitoring)**
* 📡 Real-time data streaming (IoT sensors)
* 📲 Alert system (SMS / Email / Dashboard notifications)
* ☁️ Cloud deployment (AWS / Azure / GCP)
* 🤖 Deep Learning models for complex pattern detection

---

## 🧪 Use Cases

* Manufacturing Plants
* Construction Sites
* Chemical Industries
* Smart Factories (Industry 4.0)

---

## 👨‍💻 Author

**Manoj Royal**
B.Tech CSE (AI)
Focused on building real-world AI solutions

---

## ⭐ Support

If you find this project useful:

* ⭐ Star this repository
* 🍴 Fork it
* 📢 Share with others

---

## 📜 License

This project is open-source and available under the MIT License.

out put screenshorts

<img width="1607" height="664" alt="Screenshot 2026-04-06 142351 - Copy" src="https://github.com/user-attachments/assets/ed6b865f-1544-4810-8ad9-d8bd320399c0" />
<img width="1820" height="734" alt="Screenshot 2026-04-06 142340" src="https://github.com/user-attachments/assets/cfa74f65-fa89-41cf-a3a9-700720b338ea" />
<img width="1376" height="740" alt="Screenshot 2026-04-06 142326" src="https://github.com/user-attachments/assets/f85c5747-b27e-436d-8f8e-ad81e21a981e" />
<img width="1309" height="831" alt="Screenshot 2026-04-06 142233" src="https://github.com/user-attachments/assets/32773379-cd43-4619-8ca9-f7da406e4474" />
<img width="923" height="848" alt="Screenshot 2026-04-06 142219" src="https://github.com/user-attachments/assets/af7772ec-5110-4824-be0a-380f985728b5" />
<img width="1097" height="549" alt="Screenshot 2026-04-06 142206" src="https://github.com/user-attachments/assets/8ae05b41-5066-4490-9099-4813045dcc2f" />
<img width="1204" height="775" alt="Screenshot 2026-04-06 142157" src="https://github.com/user-attachments/assets/fdd5ee82-6cb3-42ff-a3e1-26b321deaa5a" />
<img width="1226" height="693" alt="Screenshot 2026-04-06 142141" src="https://github.com/user-attachments/assets/7b76dac5-ffca-482a-b195-dde0974b26ae" />
<img width="1920" height="1080" alt="Screenshot (22)" src="https://github.com/user-attachments/assets/ffd486e6-e775-4b9b-92f3-8d36d79db387" />
<img width="1436" height="831" alt="Screenshot 2026-04-06 142131" src="https://github.com/user-attachments/assets/4528d395-1296-4a97-af54-1f3f48339b21" />
<img width="1529" height="710" alt="Screenshot 2026-04-06 142107" src="https://github.com/user-attachments/assets/7a13fcd9-43e5-412d-ac9a-ace4215afa2c" />
<img width="1880" height="823" alt="Screenshot 2026-04-06 141946" src="https://github.com/user-attachments/assets/1944156e-6c26-4749-bd1e-e5ee14deb26a" />
<img width="1910" height="901" alt="Screenshot 2026-04-06 141859" src="https://github.com/user-attachments/assets/24705782-2664-436f-9cc9-6217a490798a" />
<img width="1909" height="893" alt="Screenshot 2026-04-06 141926" src="https://github.com/user-attachments/assets/7bd2fb5e-76c9-4cf6-89df-aab1c4c4cd2f" />



