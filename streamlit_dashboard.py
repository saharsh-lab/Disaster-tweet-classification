import sys
import streamlit as st

# Ensure page config is called first before any other Streamlit component
st.set_page_config(page_title="Disaster Response Classifier Dashboard", layout="wide")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300000)  # Refresh every 300,000 ms = 5 minutes
except Exception:
    pass

from automated_dashboard import AutomatedDashboard

dashboard = AutomatedDashboard()
dashboard.create_streamlit_app() 