import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

class AutomatedDataCollector:
    def __init__(self, data_directory="datasets"):
        self.data_directory = data_directory
        self.datasets = []
        self.load_available_datasets()
    
    def load_available_datasets(self):
        """Load all available datasets from the data directory"""
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            print(f"Created {self.data_directory} directory. Please add your datasets there.")
            return
        
        # Look for CSV files in the data directory
        csv_files = glob.glob(os.path.join(self.data_directory, "*.csv"))
        self.datasets = csv_files
        print(f"Found {len(csv_files)} datasets: {[os.path.basename(f) for f in csv_files]}")
    
    def collect_new_data(self, max_tweets=1000):
        """Collect new data from multiple sources"""
        all_data = []
        
        for dataset_path in self.datasets:
            try:
                data = pd.read_csv(dataset_path)
                print(f"Loaded {len(data)} tweets from {os.path.basename(dataset_path)}")
                all_data.append(data)
            except Exception as e:
                print(f"Error loading {dataset_path}: {e}")
                continue
        
        if not all_data:
            print("No datasets found. Using sample data...")
            return self.generate_sample_data(max_tweets)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates based on text content
        combined_data = combined_data.drop_duplicates(subset=['text'], keep='first')
        
        # Limit the number of tweets to process
        if len(combined_data) > max_tweets:
            combined_data = combined_data.sample(n=max_tweets, random_state=42)
        
        print(f"Total unique tweets to process: {len(combined_data)}")
        return combined_data
    
    def filter_recent_data(self, data, hours=24):
        """Filter data from last N hours (if timestamp column exists)"""
        if 'timestamp' in data.columns:
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                recent_data = data[data['timestamp'] > datetime.now() - timedelta(hours=hours)]
                print(f"Filtered to {len(recent_data)} recent tweets (last {hours} hours)")
                return recent_data
            except:
                print("Could not parse timestamps, returning all data")
                return data
        else:
            print("No timestamp column found, returning all data")
            return data
    
    def generate_sample_data(self, num_tweets=100):
        """Generate sample disaster-related tweets for testing"""
        sample_tweets = [
            "Need food and water urgently in Chennai!",
            "Shelter required for family of 4 in Mumbai",
            "Medical help needed for elderly person",
            "Rescue team needed in flooded area",
            "Clothes and blankets needed for children",
            "The weather is nice today",
            "Going to the grocery store",
            "Having dinner with friends",
            "Need immediate medical assistance",
            "Water supply cut off in our area",
            "Food shortage in the neighborhood",
            "Emergency shelter needed for displaced families",
            "Rescue operations ongoing in the affected area",
            "Medical supplies running low",
            "Clothing donations needed for victims",
            "Just finished my homework",
            "Watching a movie tonight",
            "Going for a walk in the park",
            "Cooking dinner for the family",
            "Reading a good book"
        ]
        
        # Repeat and shuffle to get desired number of tweets
        repeated_tweets = sample_tweets * (num_tweets // len(sample_tweets) + 1)
        selected_tweets = repeated_tweets[:num_tweets]
        
        # Create DataFrame with labels (1 for disaster requests, 0 for normal tweets)
        disaster_keywords = ['need', 'urgent', 'help', 'emergency', 'rescue', 'shelter', 'medical', 'food', 'water', 'clothes']
        
        labels = []
        for tweet in selected_tweets:
            if any(keyword in tweet.lower() for keyword in disaster_keywords):
                labels.append(1)
            else:
                labels.append(0)
        
        sample_data = pd.DataFrame({
            'text': selected_tweets,
            'label': labels,
            'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 24)) for _ in range(num_tweets)]
        })
        
        print(f"Generated {num_tweets} sample tweets for testing")
        return sample_data
    
    def save_processed_data(self, data, filename="processed_data.csv"):
        """Save processed data for later use"""
        data.to_csv(filename, index=False)
        print(f"Saved processed data to {filename}")
    
    def get_data_statistics(self, data):
        """Get statistics about the collected data"""
        stats = {
            'total_tweets': len(data),
            'disaster_requests': len(data[data['label'] == 1]) if 'label' in data.columns else 'Unknown',
            'normal_tweets': len(data[data['label'] == 0]) if 'label' in data.columns else 'Unknown',
            'datasets_used': len(self.datasets)
        }
        return stats 