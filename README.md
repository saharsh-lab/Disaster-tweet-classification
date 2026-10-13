<p align="center">
  <img src="assets/flarewatch-logo.svg" alt="Flarewatch — Crisis Signal Intelligence" width="420">
</p>

# Flarewatch — Crisis Signal Intelligence

> *See every distress signal before it fades.* Flarewatch is an AI disaster intelligence platform that classifies social-media crisis signals with a bidirectional LSTM and extracts the humanitarian resources people are asking for.

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

### 🧠 Models (pure NumPy at inference time)
- **Request classifier — Bidirectional LSTM (v2):** decides whether a message is someone asking for help. Trained on 21k real disaster messages labelled for requests; **88.2% accuracy, 71.3% request F1, 89.7% ROC-AUC** on the held-out test set (the original tornado-trained v1 scored 32.3% / 30.6% on the same test). See [MODEL_CARD.md](MODEL_CARD.md).
- **Needs model — multi-label TF-IDF + logistic regression:** tags water, food, shelter, clothing, medical aid and rescue, several per message. **75.8% micro-F1 on real requests vs. 53.7%** for the original keyword rules.
- **Word-level explanations:** occlusion — each word is removed in turn and the change in P(request) is reported, so the UI shows which words actually drove a decision.
- **Serverless-friendly:** the Bi-LSTM and the needs model run as NumPy forward passes (no TensorFlow on the server, verified against Keras to 5 × 10⁻⁷). The Keras `.h5` + `tokenizer.pkl` path is still used by the batch pipeline and as a local fallback.

### 🌍 Live signals & locations
- **Live feed (`/api/live`):** public Mastodon hashtag timelines (#flood, #earthquake, #wildfire, …) — no API key — plus Bluesky search when credentials are configured. Bot accounts and non-English posts are skipped, everything is classified on arrival and cached for 3 minutes.
- **Place extraction:** a GeoNames gazetteer (34k cities, regions and countries) finds places named in a message, with context-based disambiguation ("Alabama and Georgia" → the US state). Request locations are plotted on the globe and listed on the Analytics tab.

### 🏷️ Humanitarian Resource Classification
- 💧 **Water & Sanitation** · 🏥 **Medical Aid** · 🏕️ **Emergency Shelter** · 🍞 **Food & Rations** · 🦺 **Search & Rescue** · 👕 **Clothing & Warmth**

### 💻 Web Application
- **Warm Charcoal & Sand design system**: Charcoal surfaces, sand accents, soft-white type and a single muted-red alert colour, with matching **Light** and **Dark** themes (plus System sync). Instrument Serif display type, Inter UI text and IBM Plex Mono figures.
- **Restrained motion**: Quiet page transitions, count-up metrics and a single 3D element — a slowly rotating, draggable dot globe marking places named in requests and live disaster posts. Everything pauses with the built-in *Reduce motion* switch or the OS setting.
- **Interactive charts (Chart.js)**: Request split doughnut, confidence histogram with the 0.50 decision threshold, resource-demand polar chart, training-prior vs. stream drift check, toggleable training curves and a per-class precision/recall/F1 radar — all driven by the stats API and `model_metrics.json`.
- **Classification console**: One-click examples, probability gauge, word-influence explanation, multi-need badges, places mentioned, measured round-trip latency and notifications (`⌘/Ctrl + Enter` to run).
- **Model transparency**: v1 vs. baseline vs. v2 comparison, sanity probes, needs-model vs. keyword-rules table and the full confusion matrix on the Model tab.
- **Accessible by design**: WCAG-AA text contrast in both themes, visible focus rings, skip link, ARIA roles/states for navigation, toggles, charts and the inspection drawer, and keyboard support throughout.
- **Progressive enhancement**: AJAX calls to the JSON API, with a plain Flask form POST fallback rendered server-side.
- **Responsive**: Off-canvas navigation and single-column layouts down to 360 px; deep links per tab (e.g. `/#model`).

### 📊 Streamlit Monitoring & Analytics Dashboard
- Flarewatch-themed console with light and dark themes (`.streamlit/config.toml`), matching Plotly styling chosen from the viewer's active theme, and tabs for Overview, Signal stream and Export.
- Live feed telemetry tracking request vs. informational/casual post ratios using Plotly, plus a confidence histogram split by class.
- Humanitarian resource breakdown charts and distribution analytics.
- Historical stream of 8,000 batch classifications (1,729 distinct tweets, re-scored with the current models).
- Instant CSV and JSON summary export capabilities.

### ⚙️ Automated Batch Processing & Scheduling
- Automated periodic ingestion pipeline (`main_automation.py`) with continuous cron or single-batch execution.
- Generates HTML executive summaries and structured audit logs.

---

## 📁 Repository Structure

```text
.
├── flarewatch/                  # Shared package used by every entry point
│   ├── inference.py             #   NumPy Bi-LSTM, needs model, occlusion explanations
│   ├── geo.py                   #   Gazetteer place extraction
│   ├── live.py                  #   Mastodon / Bluesky live feed with caching
│   └── data/gazetteer.json      #   Built from GeoNames (CC BY 4.0)
├── api/                         # Vercel serverless function (index.py) + generated copies of the files below
├── templates/index.html         # Web application UI
├── predict_api.py               # Local Flask server & REST API
├── data_service.py              # Stats, stream and dataset queries
├── streamlit_dashboard.py       # Streamlit dashboard launcher (automated_dashboard.py renders it)
├── automated_classifier.py      # Batch classifier (Keras) used by main_automation.py
├── train_request_model.py       # Trains + evaluates the Bi-LSTM request classifier
├── train_needs_model.py         # Trains + evaluates the multi-label needs model
├── rescore_results.py           # Re-scores the stored stream with the deployed models
├── scripts/
│   ├── build_gazetteer.py       # Builds flarewatch/data/gazetteer.json
│   └── sync_vercel_bundle.py    # Copies canonical files into api/ (--check in CI)
├── tests/                       # unittest suite (models, geo, live feed, API, bundle sync)
├── model_weights.npz            # Request classifier weights (NumPy)
├── word_index.json              # Vocabulary
├── needs_model.npz              # Needs model weights
├── model_metrics.json           # Evaluation report read by the dashboard
├── disaster_rnn_model.h5        # Keras version of the request classifier
├── tokenizer.pkl                # Keras tokenizer
├── classification_results.csv   # Stored stream (8,000 rows, 1,729 distinct tweets)
├── dataset_cleaned.csv          # Training corpus shown on the Dataset tab
├── train_model.ipynb            # Original v1 notebook (kept for reference)
├── MODEL_CARD.md                # Data, metrics, limitations, how to reproduce
├── .github/workflows/ci.yml     # Tests on every push / pull request
└── vercel.json                  # Vercel routing
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

### Live feed configuration (optional)

| Variable | Default | Purpose |
|---|---|---|
| `FLAREWATCH_MASTODON_INSTANCE` | `mastodon.social` | Instance whose public hashtag timelines are read |
| `FLAREWATCH_LIVE_TAGS` | flood, earthquake, wildfire, hurricane, cyclone, tornado, disasterrelief, evacuation | Comma-separated hashtags |
| `FLAREWATCH_LIVE_TTL` | `180` | Cache lifetime in seconds |
| `FLAREWATCH_LIVE_INCLUDE_BOTS` | unset | `1` to keep automated accounts |
| `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD` | unset | Enable Bluesky search (free app password from Bluesky settings) |

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

#### Response (`200 OK`), abridged (`explanation` shows the 3 most influential words; `tokens` omitted):
```json
{
  "is_request": true,
  "probability": 0.8128,
  "confidence_percent": 81.3,
  "resource": "rescue",
  "resources": [
    {
      "name": "rescue",
      "probability": 0.9806,
      "detected": true,
      "source": "model",
      "evidence": [
        "rescue",
        "trapped",
        "rooftops"
      ]
    },
    {
      "name": "water",
      "probability": 0.8136,
      "detected": true,
      "source": "model",
      "evidence": [
        "water",
        "drinking water",
        "drinking"
      ]
    }
  ],
  "locations": [],
  "explanation": [
    {
      "token": "need",
      "weight": 0.4682,
      "in_vocab": true
    },
    {
      "token": "sector",
      "weight": 0.1753,
      "in_vocab": true
    },
    {
      "token": "water",
      "weight": 0.1314,
      "in_vocab": true
    }
  ],
  "cleaned": "urgent flash flooding sector people trapped rooftops need immediate rescue boats clean drinking water",
  "status": "High confidence"
}
```
`resource` (single primary need) is kept for backward compatibility; `resources`, `locations` and `explanation` are additive.

### `GET /api/live`
Recent public posts from Mastodon (and Bluesky if configured), each classified with `is_request`, `probability`, `resources` and `locations`. Responses are cached for `FLAREWATCH_LIVE_TTL` seconds; `status` is `ok`, `partial`, `stale` (serving the last good snapshot) or `unavailable`.

### Other endpoints
`GET /api/stats` · `GET /api/tweets` · `GET /api/dataset` · `GET /api/model-info` · `GET /api/health` · `GET /export/{results.csv|dataset.csv|stats.json}`

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

## 🔬 Models, Training & Evaluation

Full details — data, metrics, limitations and reproduction steps — are in **[MODEL_CARD.md](MODEL_CARD.md)**.

| Request classifier (same test set) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| v1 Bi-LSTM (tornado labels) | 32.3% | 19.0% | 78.6% | 30.6% | 46.4% |
| TF-IDF + logistic regression baseline | 86.7% | 62.3% | 75.2% | 68.1% | 88.9% |
| **v2 Bi-LSTM (deployed)** | **88.2%** | **66.2%** | **77.4%** | **71.3%** | **89.7%** |

```bash
pip install -r requirements-full.txt
python train_request_model.py --install   # Bi-LSTM request classifier
python train_needs_model.py --install     # multi-label needs model
python rescore_results.py                 # update the stored stream
python scripts/sync_vercel_bundle.py      # refresh api/ for Vercel
```

## ✅ Tests & CI

```bash
python -m unittest discover -s tests -t .
```
33 tests cover preprocessing, NumPy-vs-training parity, explanations, the needs model, place extraction, the live feed (network mocked), every API route including the Vercel rewrite dispatcher, and that `api/` is in sync with the repository root. GitHub Actions runs them on every push and pull request with only the serving dependencies installed.

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

Third-party data: Disaster Response Messages (Figure Eight / Appen; see the licence note in [MODEL_CARD.md](MODEL_CARD.md)) and place data from [GeoNames](https://www.geonames.org) (CC BY 4.0).
