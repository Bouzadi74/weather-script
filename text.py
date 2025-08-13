"""
Module d'extraction et de traitement de texte pour les données météorologiques.
Gère l'extraction de texte depuis PDF, images, XML et le parsing des données météo.
"""

# text.py : Fonctions utilitaires pour l'extraction de texte et le parsing des données météo
# Inclut l'extraction depuis PDF, image, XML et le parsing NLP des données météo

import pdfplumber
import pytesseract
import streamlit as st
import re
from datetime import datetime
import spacy
from PIL import Image
import xml.etree.ElementTree as ET


def extract_text_from_pdf(pdf_file):
    """
    Extrait le texte d'un fichier PDF.
    
    Args:
        pdf_file: Fichier PDF à traiter
    
    Returns:
        Texte extrait ou None en cas d'erreur
    """
    try:
        text =""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Erreur PDF: {str(e)}")
        return None

def extract_text_from_image(image_file):
    """
    Extrait le texte d'une image avec OCR (Tesseract).
    
    Args:
        image_file: Fichier image à traiter
    
    Returns:
        Texte extrait ou None en cas d'erreur
    """
    try:
        image = Image.open(image_file)
        costum_config = r'--oem 3 --psm 6 -l ara+fra+eng'
        text = pytesseract.image_to_string(image, config=costum_config)
        return text
    except Exception as e:
        st.error(f"Erreur image : {str(e)}")
        return None

def extract_text_from_xml(xml_file):
    """
    Extrait le texte d'un fichier XML.
    
    Args:
        xml_file: Fichier XML à traiter
    
    Returns:
        Texte extrait ou None en cas d'erreur
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        text = ' '.join([elem.text for elem in root.iter() if elem.text])
        return text
    except Exception as e:
        return None
    

@st.cache_data(ttl=3600)
def improved_parse_weather_data(text, moroccan_cities=None, european_cities=None):
    """
    Parse et extrait les données météorologiques d'un texte.
    
    Args:
        text: Texte contenant les données météo
        moroccan_cities: Liste des villes marocaines (optionnel)
        european_cities: Liste des villes européennes (optionnel)
    
    Returns:
        Dictionnaire contenant les données météo structurées
    """
    # Extraction de la date après 'etablie le'
    date = None
    etablie_pattern = re.compile(r'etablie\s+le\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2})', re.IGNORECASE)
    etablie_match = etablie_pattern.search(text)
    if etablie_match:
        date = etablie_match.group(1)
    else:
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', text)
        date = date_match.group(0) if date_match else datetime.now().strftime('%Y-%m-%d')

    # Extraction des données ville/température/phénomène
    city_temp_pheno = []
    city_pattern = re.compile(r'([\u0600-\u06FF]{2,}|[A-Z][a-zA-Z\-]{1,})')
    temp_pattern = re.compile(r'(\d{1,3})\s*[°C]*')
    pheno_pattern = re.compile(r'(صافية|غائم|ممطر|ثلج|عاصف|Clear|Cloudy|Rain|Snow|Storm|Sunny|Fog|Mist)', re.IGNORECASE)

    # Mots à exclure (non-villes)
    non_city_words = set([
        'LE', 'LA', 'LES', 'ET', 'VENDREDI', 'SAMEDI', 'DIMANCHE', 'LUNDI', 'MARDI', 'MERCREDI', 'JEUDI',
        'JANVIER', 'FEVRIER', 'MARS', 'AVRIL', 'MAI', 'JUIN', 'JUILLET', 'AOUT', 'SEPTEMBRE', 'OCTOBRE', 'NOVEMBRE', 'DECEMBRE',
        'PAGE', 'TEMPERATURE', 'MAX', 'MIN', 'ETABLI', 'PAR', 'DE', 'DU', 'THE', 'AND', 'FRIDAY', 'SATURDAY', 'SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY',
        'JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER'
    ])

    # Traitement ligne par ligne
    for line in text.split('\n'):
        # Recherche des villes, températures et phénomènes
        city_matches = city_pattern.findall(line)
        temp_match = temp_pattern.search(line)
        pheno_match = pheno_pattern.search(line)
        temp = int(temp_match.group(1)) if temp_match else None
        pheno = pheno_match.group(1) if pheno_match else ''
        
        # Ajout seulement si ville et température trouvées
        for city in city_matches:
            if not temp and not pheno:
                continue
            # Filtrage des faux positifs
            if pheno and city.lower() == pheno.lower():
                continue
            # Exclusion des mots non-villes
            if city.upper() in non_city_words:
                continue
            city_temp_pheno.append({"name": city, "temp": temp, "condition": pheno})

    # Détection du contexte (Maroc/Europe)
    context = "General"
    if moroccan_cities and european_cities:
        moroccan_count = sum(1 for c in city_temp_pheno if c['name'] in moroccan_cities)
        european_count = sum(1 for c in city_temp_pheno if c['name'] in european_cities)
        if moroccan_count > european_count:
            context = "Moroccan cities"
        elif european_count > moroccan_count:
            context = "European cities"

    weather_data = {
        "date": date,
        "context": context,
        "regions": [
            {"name": context, "cities": city_temp_pheno}
        ]
    }
    return weather_data

