# script.py : Fonctions utilitaires pour la génération de scripts et l'intégration LLM
# Inclut les fonctions pour générer des scripts avec HuggingFace et Ollama
import requests
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def generate_script(self, weather_data, language='arabic', broadcast_type='tv_news', tone='formal', duration='medium' ):
    if not weather_data:
        return "No weather data available to generate script."
    
    templates = self.script_templates.get(language, self.script_templates['arabic'])
    
    # build the script 
    script_parts = []

    #opening
    opening = templates['opening'].get(broadcast_type, templates["opening"]["tv_news"])
    script_parts.append(opening)
    script_parts.append('\n\n')

    #weather details

    if language == 'arabic':
        script_parts.append(self.generate_arabic_weather_content(weather_data, duration))
    elif language == 'french':
        script_parts.append(self.generate_french_weather_content(weather_data, duration))
    else:
        script_parts.append(self.generate_english_weather_content(weather_data, duration))

    #closing

    closing =templates['closing'].get(tone, templates['closing']['formal'])
    script_parts.append(f'\n\n{closing}')
    return ''.join(script_parts)



def generate_script_with_ollama(prompt,model='command-r7b-arabic'):


    url ='http://localhost:11434/api/generate'
    payload ={
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        print(f"Ollama prompt: {prompt}")  # Debug: print prompt
        response = requests.post(url, json=payload, timeout =180)
        response.raise_for_status()
        result = response.json()
        print(f"Ollama raw response: {result}")  # Debug: print raw response
        script = result.get("response", "")
        if not script or script.strip() == "":
            print("Ollama returned empty script.")
            return None
        return script
    except Exception as e:
        print(f"Error generating script with Ollama: {e}")
        return None