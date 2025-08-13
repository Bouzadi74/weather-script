"""
Système de cache intelligent pour les prompts et réponses météorologiques.
Utilise la similarité sémantique pour éviter de régénérer des réponses similaires.
Améliore les performances et réduit les coûts d'API.
"""

import json
import hashlib
import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherCache:
    """
    Classe de cache intelligent pour les données météorologiques.
    Utilise des embeddings pour détecter les prompts similaires.
    """
    
    def __init__(self, cache_file: str = "weather_cache.json", similarity_threshold: float = 0.85):
        """
        Initialise le système de cache.
        
        Args:
            cache_file: Fichier de cache JSON
            similarity_threshold: Seuil de similarité pour considérer un prompt comme similaire
        """
        
        """
            cache_file: Fichier de cache JSON
            similarity_threshold: Seuil de similarité pour considérer un prompt comme similaire
        """

        self.cache_file = cache_file
        self.similarity_threshold = similarity_threshold
        self.cache_data = self._load_cache()
        self.embedding_model = None
        self._initialize_embedding_model()
        
    def _initialize_embedding_model(self):
        """Initialise le modèle d'embedding pour la similarité sémantique."""
        try:
            self.embedding_model = SentenceTransformer("Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2")
            logger.info("Modèle d'embedding initialisé pour le cache")
        except Exception as e:
            logger.warning(f"Impossible d'initialiser le modèle d'embedding: {e}")
            self.embedding_model = None
    
    def _load_cache(self) -> Dict:
        """Charge le cache depuis le fichier JSON."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"Cache chargé: {len(cache)} entrées")
                return cache
            else:
                logger.info("Création d'un nouveau cache")
                return {}
        except Exception as e:
            logger.error(f"Erreur lors du chargement du cache: {e}")
            return {}
    
    def _save_cache(self):
        """Sauvegarde le cache dans le fichier JSON."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache sauvegardé: {len(self.cache_data)} entrées")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du cache: {e}")
    
    def _generate_hash(self, prompt: str) -> str:
        """Génère un hash pour le prompt."""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Génère l'embedding d'un texte."""
        if self.embedding_model is None:
            return None
        try:
            embedding = self.embedding_model.encode([text])
            return embedding[0]
        except Exception as e:
            logger.error(f"Erreur lors de l'embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux embeddings."""
        if embedding1 is None or embedding2 is None:
            return 0.0
        try:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Erreur lors du calcul de similarité: {e}")
            return 0.0
    
    def add_to_cache(self, prompt: str, response: str, metadata: Dict = None) -> None:
        """
        Ajoute une entrée au cache.
        
        Args:
            prompt: Le prompt original
            response: La réponse générée
            metadata: Métadonnées supplémentaires (optionnel)
        """
        try:
            # Génération du hash et de l'embedding
            prompt_hash = self._generate_hash(prompt)
            prompt_embedding = self._embed_text(prompt)
            
            # Création de l'entrée de cache
            cache_entry = {
                "prompt": prompt,
                "response": response,
                "prompt_hash": prompt_hash,
                "prompt_embedding": prompt_embedding.tolist() if prompt_embedding is not None else None,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Ajout au cache
            self.cache_data[prompt_hash] = cache_entry
            
            # Sauvegarde
            self._save_cache()
            
            logger.info(f"Entrée ajoutée au cache: {prompt_hash}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout au cache: {e}")
    
    def get_from_cache(self, prompt: str) -> Optional[str]:
        """
        Récupère une réponse du cache si elle existe.
        
        Args:
            prompt: Le prompt à rechercher
            
        Returns:
            La réponse du cache si trouvée, sinon None
        """
        try:
            # Recherche exacte par hash
            prompt_hash = self._generate_hash(prompt)
            if prompt_hash in self.cache_data:
                logger.info(f"Cache hit exact: {prompt_hash}")
                return self.cache_data[prompt_hash]["response"]
            
            # Recherche par similarité sémantique
            if self.embedding_model is not None:
                query_embedding = self._embed_text(prompt)
                if query_embedding is not None:
                    best_match = self._find_similar_prompt(query_embedding)
                    if best_match:
                        logger.info(f"Cache hit par similarité: {best_match['similarity']:.3f}")
                        return best_match["response"]
            
            logger.info("Cache miss")
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche dans le cache: {e}")
            return None
    
    def _find_similar_prompt(self, query_embedding: np.ndarray) -> Optional[Dict]:
        """
        Trouve le prompt le plus similaire dans le cache.
        
        Args:
            query_embedding: L'embedding du prompt de requête
            
        Returns:
            L'entrée de cache la plus similaire si trouvée
        """
        best_match = None
        best_similarity = 0.0
        
        for cache_entry in self.cache_data.values():
            if cache_entry.get("prompt_embedding") is not None:
                cached_embedding = np.array(cache_entry["prompt_embedding"])
                similarity = self._calculate_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = {
                        "response": cache_entry["response"],
                        "similarity": similarity,
                        "prompt": cache_entry["prompt"]
                    }
        
        return best_match
    
    def get_cache_stats(self) -> Dict:
        """Retourne les statistiques du cache."""
        total_entries = len(self.cache_data)
        total_size = os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
        
        # Calcul de l'âge moyen des entrées
        ages = []
        for entry in self.cache_data.values():
            if "timestamp" in entry:
                try:
                    timestamp = datetime.fromisoformat(entry["timestamp"])
                    age = datetime.now() - timestamp
                    ages.append(age.total_seconds() / 3600)  # En heures
                except Exception:
                    continue  # Ignore les entrées avec timestamp invalide
        avg_age = sum(ages) / len(ages) if ages else 0
        
        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "average_age_hours": avg_age,
            "similarity_threshold": self.similarity_threshold
        }
    
    def clear_cache(self, older_than_days: int = None) -> int:
        """
        Nettoie le cache en supprimant les anciennes entrées.
        
        Args:
            older_than_days: Supprimer les entrées plus anciennes que X jours
            
        Returns:
            Nombre d'entrées supprimées
        """
        if older_than_days is None:
            # Suppression complète
            removed_count = len(self.cache_data)
            self.cache_data = {}
        else:
            # Suppression des anciennes entrées
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            removed_count = 0
            
            keys_to_remove = []
            for key, entry in self.cache_data.items():
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if timestamp < cutoff_date:
                    keys_to_remove.append(key)
                    removed_count += 1
            
            for key in keys_to_remove:
                del self.cache_data[key]
        
        self._save_cache()
        logger.info(f"Cache nettoyé: {removed_count} entrées supprimées")
        return removed_count
    
    def search_cache(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Recherche dans le cache par similarité sémantique.
        
        Args:
            query: Requête de recherche
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des entrées les plus similaires
        """
        if self.embedding_model is None:
            return []
        
        query_embedding = self._embed_text(query)
        if query_embedding is None:
            return []
        
        results = []
        for entry in self.cache_data.values():
            if entry.get("prompt_embedding") is not None:
                cached_embedding = np.array(entry["prompt_embedding"])
                similarity = self._calculate_similarity(query_embedding, cached_embedding)
                
                results.append({
                    "prompt": entry["prompt"],
                    "response": entry["response"],
                    "similarity": similarity,
                    "timestamp": entry["timestamp"]
                })
        
        # Tri par similarité décroissante
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]


# Fonction utilitaire pour intégration facile
def get_cached_response(prompt: str, cache_file: str = "weather_cache.json") -> Optional[str]:
    """
    Fonction utilitaire pour récupérer une réponse du cache.
    
    Args:
        prompt: Le prompt à rechercher
        cache_file: Fichier de cache à utiliser
        
    Returns:
        La réponse du cache si trouvée, sinon None
    """
    cache = WeatherCache(cache_file)
    return cache.get_from_cache(prompt)


def add_to_cache(prompt: str, response: str, cache_file: str = "weather_cache.json", metadata: Dict = None) -> None:
    """
    Fonction utilitaire pour ajouter une entrée au cache.
    
    Args:
        prompt: Le prompt original
        response: La réponse générée
        cache_file: Fichier de cache à utiliser
        metadata: Métadonnées supplémentaires
    """
    cache = WeatherCache(cache_file)
    cache.add_to_cache(prompt, response, metadata)


# Test du système de cache
if __name__ == "__main__":
    # Test du système de cache
    cache = WeatherCache()
    
    # Ajout d'entrées de test
    test_prompts = [
        "أنت مذيع نشرة جوية محترف. Casablanca 22 Ciel_Clair",
        "أنت مذيع نشرة جوية محترف. Rabat 21 Peu_Nuageux",
        "أنت مذيع نشرة جوية محترف. Marrakech 28 Ciel_Clair"
    ]
    
    test_responses = [
        "أسعد الله أوقاتكم بكل خير، إليكم النشرة الجوية...",
        "أهلاً ومرحباً بكم في نشرتنا الجوية...",
        "نرحب بكم في نشرتنا الجوية لهذا اليوم..."
    ]
    
    for prompt, response in zip(test_prompts, test_responses):
        cache.add_to_cache(prompt, response)
    
    # Test de récupération
    test_query = "أنت مذيع نشرة جوية محترف. Casablanca 22 Ciel_Clair"
    cached_response = cache.get_from_cache(test_query)
    
    if cached_response:
        print(f"✅ Réponse trouvée dans le cache: {cached_response[:50]}...")
    else:
        print("❌ Aucune réponse trouvée dans le cache")
    
    # Statistiques du cache
    stats = cache.get_cache_stats()
    print(f"📊 Statistiques du cache: {stats}") 