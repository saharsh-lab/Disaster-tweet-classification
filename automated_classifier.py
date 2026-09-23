import pickle
import numpy as np
import pandas as pd
import re
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class AutomatedClassifier:
    def __init__(self, model_path="disaster_rnn_model.h5", tokenizer_path="tokenizer.pkl"):
        """Initialize the classifier with pre-trained model and tokenizer"""
        try:
            self.model = load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
        
        try:
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            self.tokenizer = None
        
        self.MAX_LEN = 30
        
        # Define stopwords for text cleaning
        self.STOPWORDS = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'to', 'from', 'in', 'on', 'for', 'of', 'at',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", '', text)  # Remove URLs
        text = re.sub(r"@\w+|#\w+", '', text)       # Remove mentions/hashtags
        text = re.sub(r"[^a-z\s]", '', text)        # Remove punctuation and numbers
        words = text.split()
        words = [w for w in words if w not in self.STOPWORDS]
        return ' '.join(words)
    
    def extract_resource(self, text):
        """Extract the type of resource being requested"""
        if not isinstance(text, str):
            return "unspecified resource"
        
        text = text.lower()
        if "food" in text:
            return "food"
        elif "water" in text:
            return "water"
        elif "shelter" in text or "accommodation" in text:
            return "shelter"
        elif "medical" in text or "medicine" in text or "doctor" in text:
            return "medical aid"
        elif "clothes" in text or "clothing" in text:
            return "clothing"
        elif "rescue" in text:
            return "rescue"
        else:
            return "unspecified resource"
    
    def predict_single(self, tweet):
        """Predict for a single tweet"""
        if self.model is None or self.tokenizer is None:
            return {
                'tweet': tweet,
                'is_request': False,
                'confidence': 0.0,
                'resource': 'model_not_loaded',
                'error': 'Model or tokenizer not loaded'
            }
        
        try:
            # Clean and preprocess
            cleaned = self.clean_text(tweet)
            
            # Tokenize and pad
            seq = self.tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=self.MAX_LEN, padding='post')
            
            # Predict
            prediction = self.model.predict(padded, verbose=0)[0][0]
            
            # Determine if it's a request
            is_request = prediction >= 0.5
            
            # Extract resource type
            resource = self.extract_resource(tweet) if is_request else "not_applicable"
            
            return {
                'tweet': tweet,
                'cleaned_text': cleaned,
                'is_request': is_request,
                'confidence': float(prediction),
                'resource': resource,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'tweet': tweet,
                'is_request': False,
                'confidence': 0.0,
                'resource': 'error',
                'error': str(e)
            }
    
    def classify_batch(self, tweets, batch_size=32):
        """Classify a batch of tweets efficiently"""
        if self.model is None or self.tokenizer is None:
            print("❌ Model or tokenizer not loaded")
            return []
        
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(tweets), batch_size):
            batch_tweets = tweets[i:i + batch_size]
            batch_results = []
            
            try:
                # Clean all tweets in batch
                cleaned_texts = [self.clean_text(tweet) for tweet in batch_tweets]
                
                # Tokenize and pad batch
                sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
                padded_sequences = pad_sequences(sequences, maxlen=self.MAX_LEN, padding='post')
                
                # Predict batch
                predictions = self.model.predict(padded_sequences, verbose=0)
                
                # Process results
                for j, (tweet, prediction) in enumerate(zip(batch_tweets, predictions)):
                    confidence = float(prediction[0])
                    is_request = confidence >= 0.5
                    resource = self.extract_resource(tweet) if is_request else "not_applicable"
                    
                    result = {
                        'tweet': tweet,
                        'cleaned_text': cleaned_texts[j],
                        'is_request': is_request,
                        'confidence': confidence,
                        'resource': resource,
                        'timestamp': datetime.now()
                    }
                    batch_results.append(result)
                
                results.extend(batch_results)
                print(f"✅ Processed batch {i//batch_size + 1}/{(len(tweets) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"❌ Error processing batch: {e}")
                # Fallback to individual processing
                for tweet in batch_tweets:
                    result = self.predict_single(tweet)
                    results.append(result)
        
        return results
    
    def classify_dataframe(self, df, text_column='text'):
        """Classify tweets from a pandas DataFrame"""
        if text_column not in df.columns:
            print(f"❌ Column '{text_column}' not found in DataFrame")
            return pd.DataFrame()
        
        tweets = df[text_column].tolist()
        results = self.classify_batch(tweets)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original DataFrame if needed
        if len(results_df) == len(df):
            results_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        return results_df
    
    def get_classification_statistics(self, results):
        """Get statistics about classification results"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        stats = {
            'total_tweets': len(df),
            'requests_detected': len(df[df['is_request'] == True]),
            'normal_tweets': len(df[df['is_request'] == False]),
            'avg_confidence': df['confidence'].mean(),
            'resource_breakdown': df[df['is_request'] == True]['resource'].value_counts().to_dict(),
            'high_confidence_requests': len(df[(df['is_request'] == True) & (df['confidence'] > 0.8)]),
            'low_confidence_requests': len(df[(df['is_request'] == True) & (df['confidence'] < 0.7)])
        }
        
        return stats
    
    def save_results(self, results, filename="classification_results.csv"):
        """Save classification results to CSV"""
        if not results:
            print("❌ No results to save")
            return
        
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"✅ Results saved to {filename}")
    
    def print_summary(self, results):
        """Print a summary of classification results"""
        stats = self.get_classification_statistics(results)
        
        print("\n" + "="*50)
        print("CLASSIFICATION SUMMARY")
        print("="*50)
        print(f"Total tweets processed: {stats.get('total_tweets', 0)}")
        print(f"Disaster requests detected: {stats.get('requests_detected', 0)}")
        print(f"Normal tweets: {stats.get('normal_tweets', 0)}")
        print(f"Average confidence: {stats.get('avg_confidence', 0):.3f}")
        print(f"High confidence requests (>80%): {stats.get('high_confidence_requests', 0)}")
        print(f"Low confidence requests (<70%): {stats.get('low_confidence_requests', 0)}")
        
        if stats.get('resource_breakdown'):
            print("\nResource Breakdown:")
            for resource, count in stats['resource_breakdown'].items():
                print(f"  {resource}: {count}")
        
        print("="*50) 