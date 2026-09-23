#!/usr/bin/env python3
"""
Main Automation Script for Disaster Response Tweet Classifier
This script automates the entire process of collecting, classifying, and monitoring disaster-related tweets.
"""

import schedule
import time
import os
import sys
from datetime import datetime
import argparse

# Import our custom modules
from automated_data_collector import AutomatedDataCollector
from automated_classifier import AutomatedClassifier
from automated_dashboard import AutomatedDashboard

class DisasterResponseAutomation:
    def __init__(self, config=None):
        """Initialize the automation system"""
        self.config = config or {
            'data_directory': 'datasets',
            'max_tweets_per_batch': 1000,
            'batch_size': 32,
            'schedule_interval': 10,  # minutes
            'results_file': 'classification_results.csv',
            'log_file': 'automation_log.txt'
        }
        
        # Initialize components
        print("🚀 Initializing Disaster Response Automation System...")
        
        self.data_collector = AutomatedDataCollector(self.config['data_directory'])
        self.classifier = AutomatedClassifier()
        self.dashboard = AutomatedDashboard(self.config['results_file'])
        
        # Statistics tracking
        self.total_processed = 0
        self.total_requests = 0
        self.start_time = datetime.now()
        
        print("✅ Automation system initialized successfully!")
    
    def log_message(self, message):
        """Log messages to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        # Write to log file
        with open(self.config['log_file'], 'a') as f:
            f.write(log_entry + '\n')
    
    def run_automation_cycle(self):
        """Run one complete automation cycle"""
        try:
            self.log_message("🔄 Starting automation cycle...")
            
            # Step 1: Collect new data
            self.log_message("📥 Collecting data...")
            new_data = self.data_collector.collect_new_data(self.config['max_tweets_per_batch'])
            
            if len(new_data) == 0:
                self.log_message("ℹ️ No new data to process")
                return
            
            # Step 2: Classify the data
            self.log_message(f"🤖 Classifying {len(new_data)} tweets...")
            
            if 'text' in new_data.columns:
                tweets = new_data['text'].tolist()
                results = self.classifier.classify_batch(tweets, self.config['batch_size'])
            else:
                self.log_message("❌ No 'text' column found in data")
                return
            
            # Step 3: Update dashboard and statistics
            self.log_message("📊 Updating dashboard...")
            self.dashboard.update_dashboard(results)
            
            # Step 4: Print summary
            self.classifier.print_summary(results)
            
            # Step 5: Update tracking statistics
            self.total_processed += len(results)
            self.total_requests += len([r for r in results if r['is_request']])
            
            # Step 6: Log cycle completion
            self.log_message(f"✅ Cycle completed! Processed {len(results)} tweets")
            
            # Step 7: Generate periodic reports
            if self.total_processed % 1000 == 0:  # Every 1000 tweets
                self.generate_periodic_report()
            
        except Exception as e:
            self.log_message(f"❌ Error in automation cycle: {str(e)}")
    
    def generate_periodic_report(self):
        """Generate periodic reports"""
        try:
            report_filename = f"automation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self.dashboard.generate_report(report_filename)
            self.log_message(f"📄 Generated periodic report: {report_filename}")
        except Exception as e:
            self.log_message(f"❌ Error generating report: {str(e)}")
    
    def print_system_status(self):
        """Print current system status"""
        runtime = datetime.now() - self.start_time
        
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)
        print(f"Runtime: {runtime}")
        print(f"Total tweets processed: {self.total_processed}")
        print(f"Total requests detected: {self.total_requests}")
        print(f"Success rate: {(self.total_requests/self.total_processed*100):.2f}%" if self.total_processed > 0 else "N/A")
        print(f"Schedule interval: {self.config['schedule_interval']} minutes")
        print(f"Data directory: {self.config['data_directory']}")
        print(f"Results file: {self.config['results_file']}")
        print("="*60)
    
    def start_automation(self):
        """Start the automated system"""
        self.log_message("🚀 Starting Disaster Response Automation System...")
        
        # Run immediately on startup
        self.run_automation_cycle()
        
        # Schedule regular runs
        schedule.every(self.config['schedule_interval']).minutes.do(self.run_automation_cycle)
        
        self.log_message(f"⏰ Scheduled to run every {self.config['schedule_interval']} minutes")
        self.log_message("🔄 Automation system is now running...")
        
        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.log_message("🛑 Automation stopped by user")
            self.print_system_status()
    
    def run_single_cycle(self):
        """Run a single automation cycle and exit"""
        self.log_message("🔄 Running single automation cycle...")
        self.run_automation_cycle()
        self.print_system_status()
    
    def run_streamlit_dashboard(self):
        """Run the Streamlit dashboard"""
        self.log_message("📊 Starting Streamlit dashboard...")
        
        # Make sure we have the latest data
        self.dashboard.load_existing_results()
        
        # Run Streamlit
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py", "--server.port", "8501"
        ])

def create_streamlit_dashboard_file():
    """Create the Streamlit dashboard file"""
    dashboard_code = '''
import streamlit as st
from automated_dashboard import AutomatedDashboard

if __name__ == "__main__":
    dashboard = AutomatedDashboard()
    dashboard.create_streamlit_app()
'''
    
    with open("streamlit_dashboard.py", "w") as f:
        f.write(dashboard_code)
    
    print("✅ Created streamlit_dashboard.py")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Disaster Response Tweet Classifier Automation")
    parser.add_argument("--mode", choices=["continuous", "single", "dashboard"], 
                       default="continuous", help="Run mode")
    parser.add_argument("--interval", type=int, default=10, 
                       help="Schedule interval in minutes (for continuous mode)")
    parser.add_argument("--max-tweets", type=int, default=1000, 
                       help="Maximum tweets to process per batch")
    parser.add_argument("--data-dir", default="datasets", 
                       help="Directory containing datasets")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'data_directory': args.data_dir,
        'max_tweets_per_batch': args.max_tweets,
        'batch_size': 32,
        'schedule_interval': args.interval,
        'results_file': 'classification_results.csv',
        'log_file': 'automation_log.txt'
    }
    
    # Create Streamlit dashboard file if needed
    if args.mode == "dashboard":
        create_streamlit_dashboard_file()
    
    # Initialize automation system
    automation = DisasterResponseAutomation(config)
    
    # Run based on mode
    if args.mode == "continuous":
        automation.start_automation()
    elif args.mode == "single":
        automation.run_single_cycle()
    elif args.mode == "dashboard":
        automation.run_streamlit_dashboard()

if __name__ == "__main__":
    main() 