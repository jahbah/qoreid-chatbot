import os
import requests
import streamlit as st

def get_data():
    # Use Streamlit secrets if available
    api_key = st.secrets["api"]["OPENAI_API_KEY"] if "api" in st.secrets else os.getenv("API_KEY")
    url = "https://api.example.com/data"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    return response.json()
