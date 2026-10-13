# Flarewatch model card

Flarewatch runs two small models, both executed in pure NumPy at inference time (no TensorFlow on the server):

| Model | Question it answers | File |
|---|---|---|
| **Request classifier** (Bi-LSTM, v2) | Is this message someone asking for help? | `model_weights.npz`, `word_index.json` |
| **Needs model** (TF-IDF + logistic regression, v1) | Which resources does it ask for? (multi-label) | `needs_model.npz` |

Place names are found with a rule-based GeoNames gazetteer (`flarewatch/data/gazetteer.json`), not a learned model.

All figures below are produced by the training scripts and stored in `model_metrics.json`; the dashboard's Model tab reads the same file.

## Intended use

- Triage support for humanitarian and emergency teams reviewing social-media or SMS messages during a disaster.
- Portfolio / research demonstration of an end-to-end NLP system.

**Not for:** automated dispatch without human review, decisions about individuals, or non-English text (the models only see English).

## Data

**Disaster Response Messages** (Figure Eight / Appen), via the Hugging Face mirror
[`community-datasets/disaster_response_messages`](https://huggingface.co/datasets/community-datasets/disaster_response_messages).
26,248 real messages (direct SMS, news and social posts) from the 2010 Haiti earthquake, the 2010 Pakistan floods,
the 2010 Chile earthquake, Superstorm Sandy (2012) and other events, each labelled for `request` and 36 categories
including water, food, shelter, clothing, medical help and search & rescue.

| Split | Messages | Requests |
|---|---|---|
| Train | 21,046 | 3,560 (16.9%) |
| Validation | 2,573 | official split |
| Test | 2,629 | official split, used only for the numbers below |

> **Licence note:** the Hugging Face mirror lists the licence as "unknown". The data was released publicly by Figure Eight
> for research; check the original terms before any commercial use. The raw data is therefore not committed to this
> repository — re-download it with the commands under *Reproduce*.

### Why v1 was replaced

The original model (`train_model.ipynb`) was trained on 3,572 tweets about a single US tornado outbreak, labelled
"about the disaster" rather than "asking for help". It also cleaned text differently at training and serving time.
On the request test set it flags almost everything that mentions a disaster:

| Model (same test set, threshold 0.50) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| v1 Bi-LSTM (tornado labels) | 32.3% | 19.0% | 78.6% | 30.6% | 46.4% |
| TF-IDF + logistic regression baseline | 86.7% | 62.3% | 75.2% | 68.1% | 88.9% |
| **v2 Bi-LSTM (deployed)** | **88.2%** | **66.2%** | **77.4%** | **71.3%** | **89.7%** |

Hand-written sanity probes (never used for training, see `train_request_model.py`): v1 5/12 correct, v2 12/12.
Example: *"Shelter required for family of 4 in Mumbai"* — v1 0.04, v2 0.77.

## Request classifier (v2)

- **Architecture:** Embedding(5,000 × 64) → Bidirectional LSTM (64 + 64) → Dropout 0.5 → Dense(32, ReLU) → Dense(1, sigmoid). Unchanged from v1, so the serverless forward pass did not change.
- **Preprocessing:** lowercase; strip URLs, @mentions, #hashtags, digits and punctuation; drop 38 stopwords; post-pad/truncate to 30 tokens. Shared by training and every serving path (`flarewatch/inference.py`).
- **Training:** Adam (lr 1e-3), batch 32, class-weighted binary cross-entropy (requests are 17% of messages), early stopping on validation loss (best epoch 2 of 4).
- **Export check:** NumPy forward pass matches Keras to 4.8 × 10⁻⁷ on 300 test messages.
- **Test results:** accuracy 88.2%, request precision 66.2%, recall 77.4%, F1 71.3%, ROC-AUC 89.7%, macro F1 82.0%.
  Confusion matrix: TN 1,933 · FP 197 · FN 113 · TP 386.

### Explanations

`/api/predict` returns a weight per word computed by **occlusion**: each word is removed in turn and the change in
P(request) is recorded. Positive weights pushed the message towards "request". This reflects what the model actually
does (it is not a keyword list), but occlusion measures one word at a time and can miss interactions between words.

## Needs model (v1)

- **Features:** TF-IDF over unigrams + bigrams of the cleaned text (8,000 terms, min document frequency 2), L2-normalised.
- **Model:** one logistic-regression output per need (water, food, shelter, clothing, medical aid, rescue), trained in NumPy with Adam, positive-class weighting and early stopping.
- **Thresholds:** per need, chosen to maximise validation F1. The original keyword rule is kept as an extra trigger only for needs where it improved validation F1 (food, shelter, medical aid, rescue).
- **Test results on the 499 real requests in the test set** (F1, model vs. original keyword rules):

| Need | Model | Keyword rules |
|---|---|---|
| Water | 78.6% | 51.1% |
| Food | 92.4% | 84.9% |
| Shelter | 79.0% | 23.2% |
| Clothing | 55.2% | 23.5% |
| Medical aid | 40.5% | 15.2% |
| Rescue | 31.6% | 36.4% |
| **Micro-F1** | **75.8%** | **53.7%** |

On all test messages micro-F1 is 67.5% vs. 51.8%. Rescue remains weak (few examples, 2.9% of training messages).

## Limitations

- **Domain shift:** training messages are 2010–2012 SMS, news and tweets, largely from Haiti and Pakistan. Accuracy on today's social media is unmeasured; the live feed is a demonstration, not an evaluation.
- **Label noise:** the dataset's `request` label is noisy (e.g. some donation appeals are labelled as requests).
- **Precision:** about one in three flagged messages is not a request at the 0.50 threshold. Raising the threshold trades recall for precision; the tier table on the Analytics tab shows the distribution.
- **Vocabulary:** 5,000 words; unknown words (many place names, slang) map to a single `<OOV>` token.
- **English only.** Non-English posts are filtered out of the live feed.
- **Places:** gazetteer matching can mis-tag surnames that are also towns (e.g. "Thornton") and only knows places with ≥ 15,000 inhabitants plus countries and first-level regions.
- **Stored stream:** `classification_results.csv` holds 8,000 rows but only 1,729 distinct tweets (repeated batch runs); counts on the dashboard include those repeats.

## Live data

`/api/live` reads public Mastodon hashtag timelines (no key) and, when `BLUESKY_HANDLE` / `BLUESKY_APP_PASSWORD` are set,
Bluesky search. Automated accounts (e.g. earthquake bots) and non-English posts are skipped; results are cached for 3 minutes.
The Bluesky path is implemented but has not been exercised with real credentials in this repository's tests.

## Reproduce

```bash
pip install -r requirements-full.txt

# Data (not committed)
mkdir -p training_data/disaster_response training_data/geonames
for s in train validation test; do
  curl -L -o training_data/disaster_response/$s.parquet \
    "https://huggingface.co/datasets/community-datasets/disaster_response_messages/resolve/main/data/$s-00000-of-00001.parquet"
done
curl -L -o training_data/geonames/cities15000.zip https://download.geonames.org/export/dump/cities15000.zip
unzip -o training_data/geonames/cities15000.zip -d training_data/geonames
curl -L -o training_data/geonames/countryInfo.txt https://download.geonames.org/export/dump/countryInfo.txt
curl -L -o training_data/geonames/admin1CodesASCII.txt https://download.geonames.org/export/dump/admin1CodesASCII.txt

python train_request_model.py --install   # request classifier -> repo root
python train_needs_model.py --install     # needs model -> repo root
python scripts/build_gazetteer.py         # places
python rescore_results.py                 # re-score the stored stream with the new models
python scripts/sync_vercel_bundle.py      # refresh the api/ copies used by Vercel
python -m unittest discover -s tests -t .
```

## Attribution

- Disaster Response Messages — Figure Eight / Appen.
- Place data — [GeoNames](https://www.geonames.org), CC BY 4.0.
