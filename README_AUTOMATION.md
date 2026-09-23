# 🚨 Disaster Response Tweet Classifier - Automated System

This is an automated version of the disaster response tweet classifier that processes datasets automatically without requiring manual input or Twitter API access.

## 🎯 What's New

- **Fully Automated**: No manual tweet input required
- **Dataset Processing**: Works with CSV datasets instead of real-time Twitter
- **Batch Processing**: Efficiently processes thousands of tweets
- **Real-time Dashboard**: Beautiful Streamlit dashboard with live statistics
- **Scheduled Automation**: Runs continuously at specified intervals
- **Comprehensive Logging**: Tracks all activities and generates reports

## 📁 Project Structure

```
disaster-project/
├── automated_data_collector.py    # Collects and processes datasets
├── automated_classifier.py        # Classifies tweets in batches
├── automated_dashboard.py         # Creates visualizations and reports
├── main_automation.py            # Main automation controller
├── requirements.txt              # Python dependencies
├── datasets/                     # Directory for your CSV datasets
├── disaster_rnn_model.h5         # Pre-trained model
├── tokenizer.pkl                 # Pre-trained tokenizer
└── README_AUTOMATION.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Datasets

Create a `datasets` folder and add your CSV files. Each CSV should have a `text` column containing the tweets:

```csv
text,label
"Need food and water urgently!",1
"The weather is nice today",0
"Shelter required for family",1
```

### 3. Run the Automation

#### Option A: Continuous Mode (Recommended)
```bash
python main_automation.py --mode continuous --interval 30
```
This runs the system continuously, processing data every 30 minutes.

#### Option B: Single Run
```bash
python main_automation.py --mode single
```
This runs one cycle and exits.

#### Option C: Dashboard Only
```bash
python main_automation.py --mode dashboard
```
This opens the Streamlit dashboard in your browser.

## 📊 Dashboard Features

The automated dashboard provides:

- **Real-time Statistics**: Total tweets, requests detected, confidence scores
- **Visual Charts**: Pie charts, bar charts, time series
- **Resource Breakdown**: What types of resources are being requested
- **Export Functionality**: Download results as CSV or JSON
- **Recent Activity**: Last 20 classification results

## ⚙️ Configuration Options

### Command Line Arguments

```bash
python main_automation.py [OPTIONS]

Options:
  --mode {continuous,single,dashboard}  Run mode (default: continuous)
  --interval INT                        Schedule interval in minutes (default: 30)
  --max-tweets INT                      Max tweets per batch (default: 1000)
  --data-dir PATH                       Dataset directory (default: datasets)
```

### Example Configurations

```bash
# Process 500 tweets every 15 minutes
python main_automation.py --mode continuous --interval 15 --max-tweets 500

# Process 2000 tweets from custom directory
python main_automation.py --mode single --max-tweets 2000 --data-dir my_datasets

# Open dashboard
python main_automation.py --mode dashboard
```

## 📈 How It Works

### 1. Data Collection
- Scans the `datasets` directory for CSV files
- Loads and combines all datasets
- Removes duplicates and limits batch size
- Generates sample data if no datasets found

### 2. Classification
- Cleans and preprocesses tweets
- Uses the pre-trained RNN model
- Classifies as disaster request or normal tweet
- Extracts resource type (food, water, shelter, etc.)

### 3. Dashboard Updates
- Updates statistics in real-time
- Creates visualizations
- Saves results to CSV
- Generates periodic reports

### 4. Automation
- Runs on schedule (every 30 minutes by default)
- Logs all activities
- Handles errors gracefully
- Generates periodic reports

## 📁 Dataset Format

Your CSV files should have this structure:

```csv
text,label,timestamp
"Need food urgently!",1,2023-12-01 10:30:00
"The weather is nice",0,2023-12-01 10:31:00
"Shelter needed",1,2023-12-01 10:32:00
```

**Required Columns:**
- `text`: The tweet content
- `label`: 1 for disaster request, 0 for normal tweet (optional)
- `timestamp`: When the tweet was posted (optional)

## 📊 Output Files

The system generates several output files:

- `classification_results.csv`: All classification results
- `automation_log.txt`: Detailed activity log
- `automation_report_YYYYMMDD_HHMMSS.html`: Periodic HTML reports
- `dashboard_report.html`: Dashboard summary report

## 🔧 Customization

### Adding New Resource Types

Edit `automated_classifier.py` and modify the `extract_resource` method:

```python
def extract_resource(self, text):
    text = text.lower()
    if "transport" in text:
        return "transport"
    elif "communication" in text:
        return "communication"
    # ... add more resource types
```

### Modifying Processing Logic

Edit `automated_data_collector.py` to customize data collection:

```python
def collect_new_data(self, max_tweets=1000):
    # Add your custom data collection logic here
    pass
```

### Custom Dashboard

Edit `automated_dashboard.py` to add new visualizations:

```python
def create_streamlit_app(self):
    # Add your custom dashboard components here
    pass
```

## 🐛 Troubleshooting

### Common Issues

1. **"No datasets found"**
   - Create a `datasets` folder
   - Add CSV files with a `text` column

2. **"Model not loaded"**
   - Ensure `disaster_rnn_model.h5` and `tokenizer.pkl` exist
   - Check file permissions

3. **"Streamlit not found"**
   - Install Streamlit: `pip install streamlit`

4. **"Memory errors"**
   - Reduce `--max-tweets` value
   - Process smaller batches

### Log Files

Check `automation_log.txt` for detailed error messages and system status.

## 📈 Performance Tips

- **Batch Size**: Use larger batch sizes (32-64) for faster processing
- **Dataset Size**: Keep individual CSV files under 10,000 rows
- **Memory**: Monitor memory usage with large datasets
- **Schedule**: Adjust interval based on your needs (15-60 minutes)

## 🔄 Integration with Existing Systems

### API Integration
```python
from automated_classifier import AutomatedClassifier

classifier = AutomatedClassifier()
result = classifier.predict_single("Need food urgently!")
print(result)
```

### Database Integration
```python
import pandas as pd
from automated_classifier import AutomatedClassifier

# Load from database
df = pd.read_sql("SELECT text FROM tweets", connection)
classifier = AutomatedClassifier()
results = classifier.classify_dataframe(df)
```

## 📞 Support

For issues or questions:
1. Check the log files
2. Review the troubleshooting section
3. Ensure all dependencies are installed
4. Verify dataset format

## 🎉 Success Stories

This automated system can be used for:
- **Emergency Response Centers**: Monitor social media during disasters
- **Humanitarian Organizations**: Track resource requests
- **Research Institutions**: Analyze disaster communication patterns
- **Government Agencies**: Automated crisis monitoring

---

**Ready to automate your disaster response monitoring? Start with a single run to test, then scale up to continuous monitoring!** 