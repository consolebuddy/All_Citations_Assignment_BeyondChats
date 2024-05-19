from CitationFinder import CitationFinder
import requests
import spacy
nlp = spacy.load("en_core_web_sm")
import json
import streamlit as st
import pandas as pd

API_URL = "https://devapi.beyondchats.com/api/get_message_with_sources"
def fetch_data(api_url):
    all_objects = []
    while True:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an error for bad status codes
        res = response.json()
        print(f"Fetching data from Page {res["data"]["current_page"]}...")
        for obj in res["data"]["data"]:
            all_objects.append(obj)
        if res["data"]["next_page_url"] is not None:
            api_url = res["data"]["next_page_url"]
        else:
            break

    return all_objects

def get_all_citations():
    all_objects = fetch_data(API_URL)
    citation_finder = CitationFinder()

    all_citations = []
    for obj in all_objects:
        id = obj['id']
        response = obj["response"]
        source = obj["source"]

        print(f"Finding Citations for Object ID: {id}...")
        citations = citation_finder.find_citations(response, source)
        all_citations.append({'id':id, 'response': response, 'citations': citations})

    return all_citations

print("\n Saving all citations with respect to each Object ID as a json file into 'all_citations.json'...")

def save_dict_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

all_citations = get_all_citations()
all_citations_dictionary = {'all_citations': all_citations}
# Save the dictionary to a JSON file
save_dict_to_json(all_citations_dictionary, "all_citations.json")

print("Saved Successfully!!!")
