# 🚨 Disaster Response Tweet Classifier & Intelligence Platform

[![Live Demo on Vercel](https://img.shields.io/badge/Vercel-Live%20App-000000.svg?style=for-the-badge&logo=vercel&logoColor=white)](https://disaster-tweet-classification.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) system and crisis response intelligence platform engineered to detect urgent humanitarian assistance requests in real time from social media streams, disaster feeds, and ground emergency reports. 

The system categorizes emergency relief needs (**Water**, **Medical Aid**, **Shelter**, **Food**, **Rescue**, **Clothing**) with high confidence, providing an interactive glassmorphic web interface, a RESTful JSON API, and a real-time analytics monitoring dashboard.

---

## 🌐 Live Web Application

The interactive classification platform is hosted and live on Vercel:
👉 **[https://disaster-tweet-classification.vercel.app](https://disaster-tweet-classification.vercel.app)**

Try entering disaster scenarios or humanitarian assistance queries directly in the web app to see real-time classification, confidence percentages, and resource requirement tags.

---

## 🌟 Key Features

### 🧠 Dual-Inference Architecture
- **Lightweight NumPy Serverless Engine (`model_weights.npz` + `word_index.json`)**:
  - Implements an ultra-fast, standalone forward-pass Bidirectional LSTM purely in vector NumPy operations.
  - Zero heavy C++/CUDA TensorFlow runtime required (<30MB total bundle footprint).
  - Powers the live Vercel deployment with fast cold-starts and low memory consumption.
- **Full Deep Learning Pipeline (`disaster_rnn_model.h5` + `tokenizer.pkl`)**:
  - Trained Bidirectional RNN / LSTM architecture evaluating temporal sequence patterns across tweet text.
  - Automatic fallback between NumPy and Keras backends depending on available environment dependencies.

### 🏷️ Humanitarian Resource Classification
Automatically extracts and tags critical emergency requirements:
- 💧 **Water & Sanitation**: Drinking water, potable water tanks, purification tablets.
- 🏥 **Medical Aid & Healthcare**: First responders, doctors, trauma kits, emergency medication.
- 🏕️ **Emergency Shelter**: Evacuation centers, temporary housing, displaced family accommodation.
- 🍞 **Food & Rations**: Emergency food packets, infant formula, non-perishable rations.
- 🦺 **Search & Rescue**: Boat rescue operations, trapped civilian extraction, airlifts.
- 👕 **Clothing & Warmth**: Blankets, protective clothing, winter essentials.

### 💻 Modern Web Application
- **Glassmorphism UI**: Dark mode interface designed with CSS backdrop-filters, responsive typography, and animated status badges.
- **One-Click Presets**: Pre-populated scenarios for quick testing (Flood & Water, Earthquake & Medical, Cyclone & Shelter, Food Crisis, Casual Non-Disaster).
- **Interactive Metric Visualizers**: Live confidence progress bars, probability scores, and dynamic categorization pills.
- **Form Persistence**: Preserves input query and classification state across submissions.

### 📊 Streamlit Monitoring & Analytics Dashboard
- Live feed telemetry tracking request vs. informational/casual post ratios using Plotly.
- Humanitarian resource breakdown charts and distribution analytics.
- Integrated historical database containing 8,000+ categorized disaster tweets.
- Instant CSV and JSON summary export capabilities.

### ⚙️ Automated Batch Processing & Scheduling
- Automated periodic ingestion pipeline (`main_automation.py`) with continuous cron or single-batch execution.
- Generates HTML executive summaries and structured audit logs.

---

## 📁 Repository Structure

```text
.
├── api/
│   └── index.py                 # Vercel Serverless WSGI entry point
├── templates/
│   └── index.html               # Glassmorphic web application UI
├── predict_api.py               # Flask Web Server & REST API endpoint
├── streamlit_dashboard.py       # Streamlit analytics dashboard launcher
├── automated_dashboard.py       # Streamlit metrics & visualization engine
├── automated_classifier.py      # Automated batch NLP processor
├── automated_data_collector.py  # Dataset collector and cleaning pipeline
├── main_automation.py           # Scheduled automation controller
├── disaster_rnn_model.h5        # Pre-trained Keras Bi-LSTM model weights
├── model_weights.npz            # Standalone NumPy weights for serverless inference
├── tokenizer.pkl                # Pickled text tokenizer vocabulary
├── word_index.json              # Lightweight JSON vocabulary for zero-dependency parsing
├── classification_results.csv   # Historical categorized dataset (8,000+ records)
├── dataset_cleaned.csv          # Cleaned training dataset
├── datasets/                    # Raw & benchmark disaster tweet sets
├── train_model.ipynb            # Model architecture, training & evaluation notebook
├── requirements.txt             # Minimal dependencies (Flask, NumPy, Gunicorn)
├── requirements-full.txt        # Full stack dependencies (TensorFlow, Streamlit, Plotly)
├── vercel.json                  # Vercel serverless deployment routing config
├── Procfile                     # Heroku / Railway / Render process definition
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/saharsh-lab/Disaster-tweet-classification.git
cd Disaster-tweet-classification
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

For the **lightweight web app and API**:
```bash
pip install -r requirements.txt
```

For the **full analytics stack** (TensorFlow + Streamlit + Plotly + Automation):
```bash
pip install -r requirements-full.txt
```

---

## 💻 Running the Applications

### Option A: Interactive Web App (Flask)

```bash
python predict_api.py
```
- Open **[http://127.0.0.1:5001](http://127.0.0.1:5001)** in your browser.
- *Port configuration*: The app runs on port `5001` by default (with automatic conflict detection for macOS AirPlay on port `5000`). You can specify any custom port:
  ```bash
  python predict_api.py --port=8080
  ```

### Option B: Streamlit Real-Time Analytics Dashboard

```bash
streamlit run streamlit_dashboard.py
```
- Open **[http://127.0.0.1:8501](http://127.0.0.1:8501)** in your browser to inspect live distribution charts, resource breakdown graphs, and tweet metrics.

### Option C: Automated Ingestion & Batch Runner

```bash
# Execute a single batch classification run
python main_automation.py --mode single

# Run continuous scheduled pipeline (runs every 15 minutes)
python main_automation.py --mode continuous --interval 15
```

---

## 🔌 REST API Documentation

The server exposes both a form handler and a dedicated JSON REST API for seamless integration with external bots, dispatch systems, and emergency dashboards.

### `POST /api/predict`
Accepts a JSON payload containing the tweet text and returns inference probabilities, request classification, and identified humanitarian resources.

#### Request Headers:
```http
Content-Type: application/json
```

#### Request Body:
```json
{
  "tweet": "Urgent! Flash flooding in Sector 4, people trapped on rooftops, need immediate rescue boats and clean drinking water!"
}
```

#### Response (`200 OK`):
```json
{
  "tweet": "Urgent! Flash flooding in Sector 4, people trapped on rooftops, need immediate rescue boats and clean drinking water!",
  "cleaned": "urgent flash flooding sector people trapped rooftops need immediate rescue boats clean drinking water",
  "is_request": true,
  "confidence_percent": 99.7,
  "probability": 0.9968,
  "resource": "water"
}
```

#### Example cURL:

**Live Production Endpoint (Vercel):**
```bash
curl -X POST https://disaster-tweet-classification.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tweet": "Severe earthquake damage downtown, multiple victims trapped under rubble needing emergency medical assistance"}'
```

**Local Endpoint:**
```bash
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tweet": "Severe earthquake damage downtown, multiple victims trapped under rubble needing emergency medical assistance"}'
```

---

## 🔬 Model Architecture & Training

The core classification model is a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network trained on thousands of disaster-related tweets:
1. **Text Preprocessing**: URL stripping, mention/hashtag cleanup, alphanumeric normalization, and custom stopword filtration.
2. **Embedding Layer**: Projects input tokens (sequence length: 30) into dense vector space.
3. **Bidirectional LSTM Layer**: Captures bidirectional contextual signals across prior and subsequent tokens.
4. **Dense & Sigmoid Activation**: Dense layers with Dropout regularization feeding into a calibrated Sigmoid output neuron estimating $P(\text{Disaster Request})$.

For complete training experiments, loss/accuracy curves, and confusion matrices, explore [train_model.ipynb](train_model.ipynb).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project (`https://github.com/saharsh-lab/Disaster-tweet-classification/fork`)
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
