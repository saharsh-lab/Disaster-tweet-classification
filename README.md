# 🚨 Disaster Response Tweet Classifier & Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) system designed to detect urgent humanitarian assistance requests in real time from social media messages and disaster reports. The system categorizes emergency relief needs (Water, Medical Aid, Shelter, Food, Rescue, Clothing) and provides both a live interactive Web UI and an automated monitoring analytics dashboard.

---

## 🌟 Key Features

- **🧠 Deep Learning Classifier**: Bidirectional RNN / LSTM model trained to distinguish between genuine disaster requests and informational/casual posts.
- **🏷️ Emergency Resource Extraction**: Automatically tags specific humanitarian aid requirements:
  - 💧 **Water & Sanitation**
  - 🏥 **Medical Aid & Health Services**
  - 🏕️ **Emergency Shelter & Accommodation**
  - 🍞 **Food & Rations**
  - 🦺 **Rescue Operations**
  - 👕 **Clothing & Essentials**
- **💻 Modern Web Application**:
  - Responsive dark glassmorphism interface with Google Fonts typography (`Inter`).
  - Interactive preset quick-test buttons for flood, earthquake, cyclone, and non-disaster scenarios.
  - Live model confidence gauge and probability breakdown.
  - RESTful JSON API (`POST /api/predict`) for third-party integrations.
- **📊 Real-time Streamlit Analytics Dashboard**:
  - Live visualization of requests vs. normal messages (Plotly).
  - Resource breakdown charts and confidence metrics.
  - Historical database of 8,000+ categorized tweets.
  - Export capabilities (CSV / JSON summary).
- **⚙️ Automated Batch Processing Pipeline**:
  - Scheduled dataset ingestion and automated reporting (`main_automation.py`).

---

## 📁 Repository Structure

```text
├── predict_api.py              # Flask Web Server & REST API endpoint
├── templates/
│   └── index.html             # Glassmorphic UI template for Web App
├── streamlit_dashboard.py      # Streamlit monitoring dashboard entry point
├── automated_dashboard.py      # Dashboard visualization & statistics logic
├── automated_classifier.py     # Batch NLP classification processor
├── automated_data_collector.py # Dataset collector and preprocessor
├── main_automation.py          # Pipeline controller (Continuous/Single/Dashboard)
├── disaster_rnn_model.h5       # Pre-trained RNN / LSTM model
├── tokenizer.pkl               # Text tokenizer vocabulary
├── classification_results.csv  # Historical classification results
├── datasets/                   # Sample disaster datasets
├── train_model.ipynb           # Model training and evaluation notebook
├── requirements.txt            # Python dependencies
├── Procfile                    # Deployment entry point (Render / Railway / Heroku)
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/disaster-tweet-classifier.git
cd disaster-tweet-classifier
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Applications

#### Option A: Run the Flask Web Application
```bash
python predict_api.py
```
Open **[http://127.0.0.1:5001](http://127.0.0.1:5001)** in your browser.

#### Option B: Run the Streamlit Analytics Dashboard
```bash
streamlit run streamlit_dashboard.py
```
Open **[http://127.0.0.1:8501](http://127.0.0.1:8501)** in your browser.

#### Option C: Run Automated Pipeline
```bash
# Run one batch and exit
python main_automation.py --mode single

# Run continuous scheduled pipeline (every 15 min)
python main_automation.py --mode continuous --interval 15
```

---

## 🔌 REST API Documentation

### Classify Tweet
- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`

#### Request Payload:
```json
{
  "tweet": "Flash flooding in Sector 4, people trapped on rooftops, need immediate rescue boats and clean drinking water!"
}
```

#### Response:
```json
{
  "tweet": "Flash flooding in Sector 4, people trapped on rooftops, need immediate rescue boats and clean drinking water!",
  "cleaned": "flash flooding sector people trapped rooftops need immediate rescue boats clean drinking water",
  "is_request": true,
  "confidence_percent": 99.7,
  "probability": 0.9968,
  "resource": "water"
}
```

---

## ☁️ Deployment Guides

### Deploying the Web App to Vercel (Recommended - 1-Click Serverless)
1. Go to **[vercel.com/new](https://vercel.com/new)** and log in with your GitHub account.
2. Under "Import Git Repository", select **`saharsh-lab/Disaster-tweet-classification`**.
3. Vercel automatically detects `vercel.json` and the serverless endpoint [`api/index.py`](api/index.py).
4. Click **Deploy**!
5. Within 60 seconds, your Web App and API will be live on a fast, global serverless URL (e.g. `https://disaster-tweet-classification.vercel.app`).

### Deploying the Streamlit Dashboard to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Select your repository, set the branch to `main`, and main file path to `streamlit_dashboard.py`.
4. Click **Deploy**.

### Deploying the Web App to Render / Railway
1. Push this repository to GitHub.
2. On Render / Railway, create a new **Web Service** linked to your repo.
3. Set the build command to `pip install -r requirements.txt`.
4. Set the start command to `gunicorn predict_api:app --bind 0.0.0.0:$PORT`.
5. Deploy.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
