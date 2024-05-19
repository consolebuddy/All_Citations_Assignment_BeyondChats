import json
import streamlit as st

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

st.set_page_config(layout="wide")

st.title('Table showing all citations with respect to each response')

st.table(load_json_file('all_citations.json')['all_citations'])