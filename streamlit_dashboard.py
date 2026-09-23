import streamlit as st
from streamlit_autorefresh import st_autorefresh
from automated_dashboard import AutomatedDashboard

st_autorefresh(interval=300000)  # Refresh every 600,000 ms = 10 minutes


if __name__ == "__main__":
    dashboard = AutomatedDashboard()
    dashboard.create_streamlit_app() 