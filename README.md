# Générateur de Scripts Météorologiques

Application web pour générer automatiquement des bulletins météorologiques en arabe, français et anglais.

##  Fonctionnalités

- **Upload de fichiers** : PDF, XML, images
- **Extraction automatique** des données météo
- **Génération de scripts** avec IA (Ollama)
- **Système RAG** pour améliorer la qualité
- **Cache intelligent** pour optimiser les performances
- **Interface web** avec Streamlit

##  Structure du Projet

```
weather-forcast-generator/
├── app.py              # Application principale Streamlit
├── cache_system.py     # Système de cache intelligent
├── rag_integration.py  # Intégration RAG
├── RAG.py             # Système RAG avec embeddings
├── script.py          # Génération de scripts
├── text.py            # Extraction de texte
├── test_rag.py        # Tests du système RAG
├── install_rag.py     # Script d'installation
├── dataset.json       # Dataset d'exemples
└── requirements.txt   # Dépendances
```

##  Installation

1. **Cloner le projet**
   ```bash
   git clone <repository>
   cd weather-forcast-generator
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Installer Ollama** (pour la génération IA)
   ```bash
   # Suivre les instructions sur ollama.ai
   ```

4. **Lancer l'application**
   ```bash
   streamlit run app.py
   ```

##  Configuration

- **Ollama** : Modèle `command-r7b-arabic:latest` requis
- **Dataset** : Fichier `dataset.json` avec exemples de scripts
- **Cache** : Fichier `weather_cache.json` créé automatiquement

##  Utilisation

1. Ouvrir l'application dans le navigateur
2. Uploader un fichier météo (PDF/XML/image)
3. Sélectionner la langue du script
4. Cliquer sur "Générer le script"
5. Le script sera généré avec l'IA

##  Tests

```bash
python test_rag.py
```

##  Technologies

- **Streamlit** : Interface web
- **Ollama** : Génération IA locale
- **Sentence Transformers** : Embeddings RAG
- **FAISS** : Indexation vectorielle
- **PyTesseract** : OCR pour images
- **PDFPlumber** : Extraction PDF



##  Licence

Projet développé pour la SNRT (Société Nationale de Radiodiffusion et de Télévision). 