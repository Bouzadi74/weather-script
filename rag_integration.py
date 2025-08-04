"""
Script d'intégration du système RAG avec l'application météorologique existante.
"""

import json
import streamlit as st
from RAG import ArabicMatryoshkaRAG, retrieve_context
from script import generate_script_with_ollama
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGWeatherGenerator:
    """
    Générateur de scripts météorologiques avec système RAG intégré.
    """
    
    def __init__(self, dataset_path: str = "dataset.json"):
        """
        Initialise le générateur avec le système RAG.
        
        Args:
            dataset_path: Chemin vers le fichier dataset JSON
        """
        self.dataset_path = dataset_path
        self.rag_system = None
        self.dataset = []
        
        # Chargement du dataset
        self._load_dataset()
        
        # Initialisation du système RAG
        self._initialize_rag()
    
    def _load_dataset(self):
        """Charge le dataset depuis le fichier JSON."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            logger.info(f"Dataset chargé: {len(self.dataset)} exemples")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dataset: {e}")
            st.error(f"Erreur lors du chargement du dataset: {e}")
    
    def _initialize_rag(self):
        """Initialise le système RAG."""
        try:
            with st.spinner("Initialisation du système RAG..."):
                self.rag_system = ArabicMatryoshkaRAG(
                    model_name="Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
                    use_faiss=True,  # Utiliser FAISS pour de meilleures performances
                    dimension=768
                )
                
                # Chargement du dataset dans le système RAG
                self.rag_system.load_dataset_from_list(self.dataset)
                
                # Construction de l'index des embeddings
                self.rag_system.build_embeddings_index()
                
            st.success("Système RAG initialisé avec succès!")
            logger.info("Système RAG initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du RAG: {e}")
            st.error(f"Erreur lors de l'initialisation du RAG: {e}")
    
    def enhance_prompt_with_rag(self, base_prompt: str, weather_data: str, file_types: dict = None, doc_type_labels: dict = None, top_k: int = 3, diversify: bool = True) -> str:
        """
        Améliore le prompt de base avec le contexte RAG avec option de diversification.
        
        Args:
            base_prompt: Le prompt de base
            weather_data: Les données météorologiques extraites
            file_types: Dictionnaire des types de fichiers sélectionnés par l'utilisateur
            doc_type_labels: Dictionnaire des labels humains pour les types de fichiers
            top_k: Nombre d'exemples similaires à récupérer
            diversify: Si True, diversifie les résultats pour éviter la répétition
            
        Returns:
            Prompt amélioré avec le contexte RAG
        """
        # 1. Récupérer les types sélectionnés par l'utilisateur
        selected_types = set()
        if file_types and doc_type_labels:
            for t in file_types.values():
                selected_types.add(doc_type_labels.get(t, t))
        # 2. Filtrer le dataset selon le type
        if selected_types:
            filtered_dataset = [ex for ex in self.dataset if any(label in ex["prompt"] for label in selected_types)]
            if not filtered_dataset:
                filtered_dataset = self.dataset  # fallback si rien trouvé
        else:
            filtered_dataset = self.dataset
        # 3. Charger ce sous-ensemble dans le RAG
        self.rag_system.load_dataset_from_list(filtered_dataset)
        self.rag_system.build_embeddings_index()
        # 4. Recherche RAG classique
        try:
            search_prompt = f"{base_prompt} {weather_data}"
            rag_context = self.rag_system.retrieve_context(search_prompt, top_k=top_k, diversify=diversify)
            if rag_context:
                context_summary = "\n".join([f"- {context}" for context in rag_context[:2]])
                enhanced_prompt = f"""{base_prompt}\n\nContexte RAG:\n{context_summary}\n\nDonnées: {weather_data}"""
                logger.info(f"Prompt amélioré avec {len(rag_context)} exemples RAG filtrés par type")
                return enhanced_prompt
            else:
                logger.warning("Aucun contexte RAG trouvé, utilisation du prompt original")
                return f"{base_prompt}\n\n{weather_data}"
        except Exception as e:
            logger.error(f"Erreur lors de l'amélioration du prompt: {e}")
            return f"{base_prompt}\n\n{weather_data}"
    
    def generate_weather_script(self, weather_data: str, language: str = "arabic") -> str:
        """
        Génère un script météorologique avec le système RAG.
        
        Args:
            weather_data: Les données météorologiques extraites
            language: Langue du script (arabic, french, english)
            
        Returns:
            Script météorologique généré
        """
        try:
            # Chargement du template de prompt selon la langue
            prompt_template = self._load_prompt_template(language)
            
            # Amélioration du prompt avec RAG
            enhanced_prompt = self.enhance_prompt_with_rag(prompt_template, weather_data)
            
            # Génération du script avec Ollama
            script = generate_script_with_ollama(enhanced_prompt)
            
            return script
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du script: {e}")
            return f"Erreur lors de la génération du script: {e}"
    
    def _load_prompt_template(self, language: str) -> str:
        """
        Charge le template de prompt selon la langue.
        
        Args:
            language: Langue du template
            
        Returns:
            Template de prompt
        """
        # Mapping des langues vers les templates
        templates = {
            "arabic": """أنت مذيع نشرة جوية محترف، تتحدث باللغة العربية الفصحى بصيغة رسمية ومناسبة للبث التلفزيوني. مهمتك هي توليد نشرة جوية طبيعية وسلسة بناءً على بيانات الطقس التي ستزود بها.""",
            "french": """Vous êtes un présentateur météo professionnel qui parle français de manière formelle et appropriée pour la diffusion télévisée. Votre tâche est de générer un bulletin météo naturel et fluide basé sur les données météorologiques qui vous seront fournies.""",
            "english": """You are a professional weather presenter who speaks English in a formal manner appropriate for television broadcasting. Your task is to generate a natural and smooth weather bulletin based on the weather data that will be provided to you."""
        }
        
        return templates.get(language.lower(), templates["arabic"])


def enhance_prompt_with_rag(base_prompt: str, weather_data: str, dataset: list, top_k: int = 3, diversify: bool = True) -> str:
    """
    Fonction utilitaire pour améliorer un prompt avec le contexte RAG avec option de diversification.
    
    Args:
        base_prompt: Le prompt de base
        weather_data: Les données météorologiques
        dataset: Le dataset d'exemples
        top_k: Nombre d'exemples à récupérer
        diversify: Si True, diversifie les résultats pour éviter la répétition
        
    Returns:
        Prompt amélioré avec le contexte RAG
    """
    try:
        # Récupération du contexte RAG avec diversification
        search_prompt = f"{base_prompt} {weather_data}"
        rag_context = retrieve_context(search_prompt, dataset, top_k=top_k, diversify=diversify)
        
        if rag_context:
            # Construction du prompt amélioré (version raccourcie)
            context_summary = "\n".join([f"- {context[:100]}..." for context in rag_context[:2]])
            enhanced_prompt = f"""{base_prompt}

Contexte RAG:
{context_summary}

Données: {weather_data}"""
            return enhanced_prompt
        else:
            return f"{base_prompt}\n\n{weather_data}"
            
    except Exception as e:
        logger.error(f"Erreur lors de l'amélioration du prompt: {e}")
        return f"{base_prompt}\n\n{weather_data}"


# Exemple d'utilisation dans l'application Streamlit
def integrate_rag_in_app():
    """
    Exemple d'intégration du RAG dans l'application Streamlit existante.
    """
    # Initialisation du générateur RAG
    rag_generator = RAGWeatherGenerator()
    
    # Exemple d'utilisation
    weather_data = """
    Ville Tmax Phenomenes
    Casablanca 22 Ciel_Clair
    Rabat 21 Peu_Nuageux
    Marrakech 28 Ciel_Clair
    """
    
    # Génération du script avec RAG
    script = rag_generator.generate_weather_script(weather_data, "arabic")
    
    return script


if __name__ == "__main__":
    # Test du système d'intégration
    print("Test du système RAG intégré...")
    
    # Chargement du dataset
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Test de la fonction utilitaire
    weather_data = "Casablanca 22 Ciel_Clair, Rabat 21 Peu_Nuageux"
    base_prompt = "أنت مذيع نشرة جوية محترف"
    
    enhanced_prompt = enhance_prompt_with_rag(base_prompt, weather_data, dataset)
    print("Prompt amélioré avec RAG:")
    print(enhanced_prompt[:500] + "...") 