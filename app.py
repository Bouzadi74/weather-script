"""
Application Streamlit pour la génération automatique de scripts météorologiques.
Permet l'upload de fichiers PDF et XML, extrait les données météo
et génère des bulletins météo en arabe, français et anglais.
"""

import streamlit as st
import json
import pdfplumber
import pymupdf4llm
import io
import base64
from PIL import Image
import pytesseract
from typing import Dict, List, Optional
import re
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
from text import extract_text_from_image, extract_text_from_pdf, extract_text_from_xml, improved_parse_weather_data
from script import generate_script_with_ollama
from rag_integration import RAGWeatherGenerator, enhance_prompt_with_rag
from cache_system import WeatherCache, get_cached_response, add_to_cache

# Couleurs du thème SNRT
SNRT_PRIMARY = "#d8d8d8"
SNRT_SECONDARY = "#131434"
SNRT_DARK = "#524848"

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Générateur de scripts",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS 
st.markdown(f"""
<style>
    .main-header {{
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, {SNRT_PRIMARY} 0%, {SNRT_DARK} 100%);
        color: {SNRT_SECONDARY};
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }}
    .weather-card {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }}
    .arabic-output {{
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        line-height: 1.8;
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-right: 4px solid #667eea;
        color: #222;
    }}
    .metrics-container {{
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }}
    .upload-section {{
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }}
    .stTextArea textarea {{
        background-color: #fff !important;
        color: #222 !important;
    }}
    textarea[data-testid="stTextArea-input"][aria-label="Votre message à Ollama"] {{
        background-color: #fff !important;
        color: #222 !important;
        border: 1.5px solid #e60000 !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }}
    .ollama-response-box {{
        background: #f8f9fa !important;
        color: #222 !important;
        border-radius: 8px !important;
        border-right: 4px solid #e60000 !important;
        padding: 8px !important;
        margin-bottom: 8px !important;
    }}
</style>
""", unsafe_allow_html=True)

# Fonction pour extraire le texte des fichiers XML (mise en cache)
@st.cache_data(ttl=3600)
def extract_text_from_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        lines = []
        date = root.attrib.get('date', None)
        for ville in root.findall('ville'):
            name = ville.attrib.get('nom', 'Inconnu')
            temp = ville.findtext('txj2', '').strip()
            pheno = ville.findtext('phenomene', '').strip()
            line = f"{name}: {temp}°C, {pheno}"
            lines.append(line)
        result = f"Date: {date}\n" if date else ""
        result += "\n".join(lines)
        return result
    except Exception as e:
        st.error(f"Erreur lors de la lecture de l'XML: {str(e)}")
        return None


@st.cache_data(ttl=60)  
def load_prompt_template(language, prompt_file="prompt_weather.txt"):
    try:
        with open(prompt_file, encoding="utf-8") as f:
            content = f.read()
        lang_map = {"arabic": "[ARABIC]", "french": "[FRENCH]", "english": "[ENGLISH]"}
        section = lang_map.get(language.lower(), "[ENGLISH]")
        esc_section = re.escape(section)
        pattern = re.compile(rf"{esc_section}\s*(.*?)\n(?=\[|$)", re.DOTALL)
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
        pattern2 = re.compile(rf"{esc_section}\s*(.*)", re.DOTALL)
        match2 = pattern2.search(content)
        if match2:
            return match2.group(1).strip()
    except Exception as e:
        st.error(f"Erreur d'affichage du prompt: {e}")
    return "Generate a weather forecast for TV news in {language} based on the following data:\n{data}"

def main():
    st.markdown(f"""
    <div class="main-header" style="background: linear-gradient(90deg, {SNRT_PRIMARY} 0%, {SNRT_DARK} 100%); color: {SNRT_SECONDARY};">
        <h1>⛅ Générateur de scripts météorologiques</h1>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.image("Logo_SNRT.PNG", use_container_width=True)

    st.sidebar.header("🎛️ Réglages")
    language = st.sidebar.selectbox(
        "Language",
        ["arabic", "french", "english"],
        format_func=lambda x: {
            "arabic": "العربية (Arabic)",
            "french": "Français (French)",
            "english": "English"
        }[x]
    )
    
    # cache
    if 'weather_cache' in st.session_state and st.session_state.weather_cache:
        st.sidebar.markdown("---")
        
        try:
            cache_stats = st.session_state.weather_cache.get_cache_stats()
            cache_status = "📦 Cache actif" if cache_stats["total_entries"] > 0 else "📭 Cache vide"
            
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.sidebar.text(cache_status)
            with col2:
                if st.sidebar.button("🗑️", help="Vider le cache"):
                    removed = st.session_state.weather_cache.clear_cache()
                    st.sidebar.success(f"✓ {removed} entrées supprimées")
                    st.rerun()
                
        except Exception as e:
            st.sidebar.text("📦 Cache actif")
            if st.sidebar.button("🗑️", help="Vider le cache"):
                try:
                    removed = st.session_state.weather_cache.clear_cache()
                    st.sidebar.success(f"✓ {removed} entrées supprimées")
                    st.rerun()
                except:
                    st.sidebar.error("Erreur cache")
    
    col1, col2, col3 = st.columns([1.2, 0.3, 1])
    #col1 pour uploader les fichiers
    with col1:
        st.header("📁 Uploader les fichiers")
        uploaded_files = st.file_uploader(
            "Téléchargez un ou plusieurs fichiers météo/marin/technique/agroclimat",
            type=['pdf', 'xml', 'png', 'jpg', 'jpeg'],
            help="Téléchargez un ou plusieurs fichiers météo/marin/technique/agroclimat",
            accept_multiple_files=True
        )
        if 'file_types' not in st.session_state:
            st.session_state.file_types = {}
            
        doc_type_labels = {
            "weather": "Prévisions météo",
            "marine": "Bulletins marine",
            "agroclimate": "Données agroclimatiques",
            "technique": "Cartes synoptiques"
        }
        
        aggregated_texts = []
        if uploaded_files:
            st.subheader("📋 Types de documents")
            st.info("Sélectionnez le type pour chaque document téléchargé")
            
            for idx, uploaded_file in enumerate(uploaded_files):
                col1, col2 = st.columns([2, 2])
                with col1:
                    st.text(f"📄 {uploaded_file.name}")
                with col2:
                    
                    if uploaded_file.name not in st.session_state.file_types:
                        st.session_state.file_types[uploaded_file.name] = "weather"
                    
                    st.session_state.file_types[uploaded_file.name] = st.selectbox(
                        "Type de document",
                        ["weather", "marine", "agroclimate", "technique"],
                        key=f"file_type_{idx}",
                        format_func=lambda x: doc_type_labels[x],
                        index=["weather", "marine", "agroclimate", "technique"].index(st.session_state.file_types[uploaded_file.name])
                    )
            
            for idx, uploaded_file in enumerate(uploaded_files):
                st.success(f"✅ Fichier téléchargé : {uploaded_file.name}")
                st.info(f"📊 Taille du fichier : {uploaded_file.size} octets")
                extracted_text = None
                if uploaded_file.type == "application/pdf":
                    with st.spinner(f"📄 Extraction du texte du PDF ({uploaded_file.name})..."):
                        extracted_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type in ["application/xml", "text/xml"]:
                    with st.spinner(f"📄 Extraction du texte de l'XML ({uploaded_file.name})..."):
                        extracted_text = extract_text_from_xml(uploaded_file)
                        if extracted_text is None:
                            extracted_text = ""
                            st.warning(f"Aucune donnée extraite du fichier XML : {uploaded_file.name}")
                elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                    with st.spinner(f"🖼️ Traitement de l'image avec OCR ({uploaded_file.name})..."):
                        image = Image.open(uploaded_file)
                        st.image(image, caption=f"Carte téléchargée : {uploaded_file.name}", use_container_width=True)
                        extracted_text = extract_text_from_image(uploaded_file)
                if extracted_text:
                    aggregated_texts.append(f"--- {uploaded_file.name} ---\n{extracted_text}")
            if aggregated_texts:
                st.session_state.extracted_text = "\n\n".join(aggregated_texts)
    
    with col2:
        st.empty()
    
    # Col
    with col3:
        st.header("🎙️ Script généré")
        
        # Initialisation du système RAG et du cache (une seule fois)
        if 'rag_generator' not in st.session_state:
            try:
                st.session_state.rag_generator = RAGWeatherGenerator("dataset.json")
                # st.success("✅ Système RAG initialisé avec succès!")  # Message supprimé
            except Exception as e:
                st.error(f"❌ Erreur lors de l'initialisation du RAG: {e}")
                st.session_state.rag_generator = None
        
        # Initialisation du système de cache
        if 'weather_cache' not in st.session_state:
            try:
                st.session_state.weather_cache = WeatherCache()
                st.success("✅ Système de cache initialisé!")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'initialisation du cache: {e}")
                st.session_state.weather_cache = None
         # col 3 pour generer le script
        if st.button("🚀 Générer le script", type="primary"):
            if hasattr(st.session_state, 'extracted_text') and st.session_state.extracted_text:
                with st.spinner("🔄 Génération du script avec cache et RAG..."):
                    prompt_template = load_prompt_template(language, "prompt_weather.txt")
                    
                    doc_types_info = "\n".join([
                        f"- {fname}: {doc_type_labels[ftype]}"
                        for fname, ftype in st.session_state.file_types.items()
                    ])
                    
                    prompt = prompt_template.replace("{document_types}", doc_types_info)
                    
                    # Verifier la cache 
                    cache_hit = False
                    if st.session_state.weather_cache:
                        try:
                            final_prompt = prompt.replace("{data}", st.session_state.extracted_text)
                            cached_response = st.session_state.weather_cache.get_from_cache(final_prompt)
                            
                            if cached_response:
                                script = cached_response
                                cache_hit = True
                                st.success("⚡ Réponse trouvée dans le cache!")
                                st.info("💡 Cette réponse a été générée précédemment pour des données similaires")
                        except Exception as e:
                            st.warning(f"⚠️ Erreur lors de la vérification du cache: {e}")
                    
                    if not cache_hit:
                        if st.session_state.rag_generator:
                            try:
                                enhanced_prompt = st.session_state.rag_generator.enhance_prompt_with_chat_template(
                                    prompt, 
                                    st.session_state.extracted_text,
                                    file_types=st.session_state.file_types,
                                    doc_type_labels=doc_type_labels,
                                    top_k=1  
                                )
                                st.info("🔍 Contexte RAG appliqué avec 1 exemple pour améliorer la génération")
                            except Exception as e:
                                st.warning(f"⚠️ Erreur RAG, utilisation du prompt standard: {e}")
                                enhanced_prompt = prompt.replace("{data}", st.session_state.extracted_text)
                        else:
                            enhanced_prompt = prompt.replace("{data}", st.session_state.extracted_text)
                        
                        script = generate_script_with_ollama(enhanced_prompt)
                        
                        if st.session_state.weather_cache and script:
                            try:
                                final_prompt = prompt.replace("{data}", st.session_state.extracted_text)
                                metadata = {
                                    "language": language,
                                    "document_types": list(st.session_state.file_types.values()),
                                    "file_count": len(st.session_state.file_types)
                                }
                                st.session_state.weather_cache.add_to_cache(final_prompt, script, metadata)
                                st.info("💾 Réponse ajoutée au cache pour les futures utilisations")
                            except Exception as e:
                                st.warning(f"⚠️ Erreur lors de l'ajout au cache: {e}")
                    
                    st.session_state.generated_script = script
                    
                    if 'ollama_chat_history' not in st.session_state:
                        st.session_state.ollama_chat_history = []
                    if st.session_state.ollama_chat_history and st.session_state.ollama_chat_history[0].get('role') == 'system':
                        st.session_state.ollama_chat_history.pop(0)
                    st.session_state.ollama_chat_history.insert(0, {
                        'role': 'system',
                        'content': f"Ceci est le dernier script généré :\n\n{script}"
                    })
            else:
                st.error("⚠️ Veuillez d'abord télécharger des données !")
        if hasattr(st.session_state, 'generated_script'):
            st.subheader("📝 Script généré")
            script = st.session_state.generated_script
            if script is None or script.strip() == "":
                st.error("❌ Aucun script n'a été généré. Veuillez vérifier vos données ou réessayer.")
            else:
                if language == 'arabic':
                    st.markdown(f"""
                    <div class="arabic-output">
                        {script.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.code(script, language="markdown")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("📋 Copier le script"):
                        st.success("Script copié dans le presse-papiers !")
                with col_btn2:
                    st.download_button(
                        label="💾 Télécharger le script",
                        data=script,
                        file_name=f"tv_script_{language}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )

    #  Ollama Chat  
    st.markdown("---")
    st.header("💬 Discuter avec le modèle Ollama")
    st.info("Pour affiner ou discuter le script généré, utilisez le chat ci-dessous avec le modèle Ollama.")
    if 'ollama_chat_history' not in st.session_state:
        st.session_state.ollama_chat_history = []
    # dernier script comme contexte
    if hasattr(st.session_state, 'generated_script') and st.session_state.generated_script:
        if not st.session_state.ollama_chat_history or st.session_state.ollama_chat_history[0].get('role') != 'system':
            st.session_state.ollama_chat_history.insert(0, {
                'role': 'system',
                'content': f"Ceci est le dernier script généré :\n\n{st.session_state.generated_script}"
            })
    chat_input = st.text_area("Votre message à Ollama", "", key="ollama_chat_input")
    if st.button("Envoyer à Ollama", key="ollama_chat_send"):
        if chat_input.strip():
            st.session_state.ollama_chat_history.append({"role": "user", "content": chat_input})
            chat_prompt = "\n".join([
                ("User: " + m["content"]) if m["role"] == "user" else ("Ollama: " + m["content"]) if m["role"] == "ollama" else ("Script: " + m["content"]) for m in st.session_state.ollama_chat_history
            ])
            ollama_response = generate_script_with_ollama(chat_prompt)
            st.session_state.ollama_chat_history.append({"role": "ollama", "content": ollama_response})
    for msg in st.session_state.ollama_chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style='background:#f4f6fb; color:#222; padding:10px 12px; border-radius:10px; margin-bottom:6px; border-left:4px solid #272A89;'>
                <strong style='color:#272A89;'>Vous :</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        elif msg["role"] == "ollama":
            st.markdown(f"""
            <div class='ollama-response-box' style='background:#eaf7ea !important; color:#222 !important; border-radius:10px !important; border-right:4px solid #009966 !important; padding:10px 12px !important; margin-bottom:6px !important;'>
                <strong style='color:#009966;'>Ollama :</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        elif msg["role"] == "system":
            st.markdown(f"""
            <div style='background:#fffbe6; color:#222; padding:10px 12px; border-radius:10px; margin-bottom:10px; border-left:4px solid #e60000;'>
                <strong style='color:#e60000;'>Script :</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {SNRT_DARK};">
        <p>Développé pour SNRT | Générateur de scripts météo & technique v1.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()