import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import json
import os

class AutomatedDashboard:
    def __init__(self, results_file="classification_results.csv"):
        self.results_file = results_file
        self.results_db = []
        self.load_existing_results()
    
    def load_existing_results(self):
        """Load existing results from file"""
        if os.path.exists(self.results_file):
            try:
                df = pd.read_csv(self.results_file)
                self.results_db = df.to_dict('records')
                print(f"✅ Loaded {len(self.results_db)} existing results")
            except Exception as e:
                print(f"❌ Error loading existing results: {e}")
    
    def update_dashboard(self, new_results):
        """Update dashboard with new classification results"""
        if new_results:
            self.results_db.extend(new_results)
            self.save_results()
            print(f"✅ Updated dashboard with {len(new_results)} new results")
    
    def save_results(self):
        """Save all results to CSV file"""
        if self.results_db:
            df = pd.DataFrame(self.results_db)
            df.to_csv(self.results_file, index=False)
    
    def get_statistics(self):
        """Generate comprehensive statistics"""
        if not self.results_db:
            return {}
        
        df = pd.DataFrame(self.results_db)
        
        # Basic statistics
        stats = {
            'total_tweets': len(df),
            'requests_detected': len(df[df['is_request'] == True]),
            'normal_tweets': len(df[df['is_request'] == False]),
            'avg_confidence': df['confidence'].mean(),
            'high_confidence_requests': len(df[(df['is_request'] == True) & (df['confidence'] > 0.8)]),
            'low_confidence_requests': len(df[(df['is_request'] == True) & (df['confidence'] < 0.7)])
        }
        
        # Resource breakdown
        if 'resource' in df.columns:
            resource_df = df[df['is_request'] == True]
            if len(resource_df) > 0:
                stats['resource_breakdown'] = resource_df['resource'].value_counts().to_dict()
            else:
                stats['resource_breakdown'] = {}
        
        # Time-based statistics
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                stats['recent_requests'] = len(df[(df['is_request'] == True) & 
                                                 (df['timestamp'] > datetime.now() - timedelta(hours=1))])
                stats['requests_last_24h'] = len(df[(df['is_request'] == True) & 
                                                   (df['timestamp'] > datetime.now() - timedelta(hours=24))])
            except:
                stats['recent_requests'] = 0
                stats['requests_last_24h'] = 0
        
        return stats
    
    def create_streamlit_app(self):
        """Create Streamlit dashboard"""
        st.set_page_config(page_title="Disaster Response Classifier Dashboard", layout="wide")
        
        st.title("Disaster Response Tweet Classifier Dashboard")
        st.markdown("---")
        
        # Load and display statistics
        stats = self.get_statistics()
        
        if not stats:
            st.warning("No data available. Please run the classifier first.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tweets", stats['total_tweets'])
        
        with col2:
            st.metric("Requests Detected", stats['requests_detected'])
        
        with col3:
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.3f}")
        
        with col4:
            st.metric("High Confidence (>80%)", stats['high_confidence_requests'])
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for request vs normal tweets
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Disaster Requests', 'Normal Tweets'],
                values=[stats['requests_detected'], stats['normal_tweets']],
                hole=0.3,
                marker_colors=['#ff6b6b', '#4ecdc4']
            )])
            fig_pie.update_layout(title="Request vs Normal Tweets Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart for resource breakdown
            if stats.get('resource_breakdown'):
                resources = list(stats['resource_breakdown'].keys())
                counts = list(stats['resource_breakdown'].values())
                
                fig_bar = px.bar(
                    x=resources, 
                    y=counts,
                    title="Resource Type Breakdown",
                    color=counts,
                    color_continuous_scale='Reds'
                )
                fig_bar.update_layout(xaxis_title="Resource Type", yaxis_title="Count")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No resource breakdown data available")
        
        # Recent activity
        st.markdown("---")
        st.subheader("Recent Activity")
        
        if self.results_db:
            df = pd.DataFrame(self.results_db)
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    recent_df = df[df['timestamp'] > datetime.now() - timedelta(hours=24)]
                    
                    if len(recent_df) > 0:
                        # Time series chart
                        recent_df['hour'] = recent_df['timestamp'].dt.hour
                        hourly_requests = recent_df[recent_df['is_request'] == True].groupby('hour').size()
                        
                        fig_time = px.line(
                            x=hourly_requests.index,
                            y=hourly_requests.values,
                            title="Requests per Hour (Last 24 Hours)",
                            labels={'x': 'Hour', 'y': 'Number of Requests'}
                        )
                        st.plotly_chart(fig_time, use_container_width=True)
                    else:
                        st.info("No recent activity in the last 24 hours")
                except:
                    st.info("Could not parse timestamp data")
        
        # Detailed results table
        st.markdown("---")
        st.subheader("Recent Classification Results")
        
        if self.results_db:
            df = pd.DataFrame(self.results_db)
            # Show random 20 results
            recent_results = df.sample(n=25)
            
            # Format for display
            display_df = recent_results[['tweet', 'is_request', 'confidence', 'resource']].copy()
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.3f}")
            display_df['related_to_disaster'] = display_df['is_request'].apply(lambda x: "Yes" if x else "No")
            display_df = display_df.drop(columns=['is_request'])
            # Reorder columns if desired
            display_df = display_df[['tweet', 'related_to_disaster', 'confidence', 'resource']]
            st.dataframe(display_df, use_container_width=True)
        
        # Export functionality
        st.markdown("---")
        st.subheader("📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Results to CSV"):
                if self.results_db:
                    df = pd.DataFrame(self.results_db)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"disaster_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("Export Statistics"):
                stats_json = json.dumps(stats, indent=2)
                st.download_button(
                    label="Download Statistics (JSON)",
                    data=stats_json,
                    file_name=f"classification_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    def generate_report(self, filename="dashboard_report.html"):
        """Generate an HTML report"""
        stats = self.get_statistics()
        
        if not stats:
            return "No data available for report"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Disaster Response Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 20px; padding: 15px; background-color: #e8f4fd; border-radius: 5px; }}
                .chart {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Disaster Response Classification Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Summary Statistics</h2>
            <div class="metric">
                <strong>Total Tweets:</strong> {stats['total_tweets']}
            </div>
            <div class="metric">
                <strong>Requests Detected:</strong> {stats['requests_detected']}
            </div>
            <div class="metric">
                <strong>Average Confidence:</strong> {stats['avg_confidence']:.3f}
            </div>
            <div class="metric">
                <strong>High Confidence Requests:</strong> {stats['high_confidence_requests']}
            </div>
        """
        
        if stats.get('resource_breakdown'):
            html_content += """
            <h2>Resource Breakdown</h2>
            <table>
                <tr><th>Resource Type</th><th>Count</th></tr>
            """
            for resource, count in stats['resource_breakdown'].items():
                html_content += f"<tr><td>{resource}</td><td>{count}</td></tr>"
            html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {filename}")
        return filename
    
    def print_summary(self):
        """Print a text summary to console"""
        stats = self.get_statistics()
        
        if not stats:
            print("No data available")
            return
        
        print("\n" + "="*60)
        print("DASHBOARD SUMMARY")
        print("="*60)
        print(f"Total tweets processed: {stats['total_tweets']}")
        print(f"Disaster requests detected: {stats['requests_detected']}")
        print(f"Normal tweets: {stats['normal_tweets']}")
        print(f"Average confidence: {stats['avg_confidence']:.3f}")
        print(f"High confidence requests (>80%): {stats['high_confidence_requests']}")
        print(f"Low confidence requests (<70%): {stats['low_confidence_requests']}")
        
        if stats.get('resource_breakdown'):
            print("\nResource Breakdown:")
            for resource, count in stats['resource_breakdown'].items():
                print(f"  {resource}: {count}")
        
        print("="*60) 