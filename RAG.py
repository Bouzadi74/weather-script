"""
Système RAG (Retrieval-Augmented Generation) pour la génération de scripts météorologiques.
Utilise les embeddings Arabic Matryoshka pour la récupération sémantique de contextes.
Améliore la qualité des scripts en fournissant des exemples pertinents.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import logging
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArabicMatryoshkaRAG:
    """
    Système RAG utilisant les embeddings Arabic Matryoshka pour la récupération sémantique.
    Permet de trouver des exemples pertinents dans un dataset de scripts météorologiques.
    """
    
    def __init__(self, 
                 model_name: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
                 max_length: int = 512,
                 use_faiss: bool = False,
                 dimension: int = 768):
        """
        Initialise le système RAG.
        
        Args:
            model_name: Nom du modèle d'embeddings Arabic Matryoshka
            max_length: Longueur maximale des textes pour l'embedding
            use_faiss: Utiliser FAISS pour l'indexation (plus rapide pour de gros datasets)
            dimension: Dimension des embeddings (peut être réduite pour la vitesse)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_faiss = use_faiss
        self.dimension = dimension
        
        # Initialisation du modèle
        logger.info(f"Chargement du modèle d'embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Stockage des embeddings et données
        self.prompt_embeddings = None
        self.dataset = []
        self.faiss_index = None
        
        logger.info("Système RAG initialisé avec succès")
    
    def load_dataset(self, dataset_path: str) -> None:
        """
        Charge le dataset depuis un fichier JSON.
        
        Args:
            dataset_path: Chemin vers le fichier JSON contenant le dataset
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            logger.info(f"Dataset chargé: {len(self.dataset)} paires prompt-réponse")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dataset: {e}")
            raise
    
    def load_dataset_from_list(self, dataset: List[Dict]) -> None:
        """
        Charge le dataset depuis une liste Python.
        
        Args:
            dataset: Liste de dictionnaires avec 'prompt' et 'response'
        """
        self.dataset = dataset
        logger.info(f"Dataset chargé depuis la liste: {len(self.dataset)} paires prompt-réponse")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Génère les embeddings pour une liste de textes.
        
        Args:
            texts: Liste de textes à encoder
            
        Returns:
            Array numpy des embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                convert_to_numpy=True
            )
            
            # Réduction de dimension si nécessaire
            if embeddings.shape[1] != self.dimension:
                if embeddings.shape[1] > self.dimension:
                    # Réduction par PCA ou sélection des premières dimensions
                    embeddings = embeddings[:, :self.dimension]
                    logger.info(f"Réduction de dimension: {embeddings.shape[1]} -> {self.dimension}")
            
            return embeddings
        except Exception as e:
            logger.error(f"Erreur lors de l'embedding: {e}")
            raise
    
    def build_embeddings_index(self) -> None:
        """
        Construit l'index des embeddings pour tous les prompts du dataset.
        """
        if not self.dataset:
            raise ValueError("Dataset non chargé. Utilisez load_dataset() ou load_dataset_from_list() d'abord.")
        
        # Extraction des prompts
        prompts = [item['prompt'] for item in self.dataset]
        
        # Génération des embeddings
        logger.info("Génération des embeddings pour tous les prompts...")
        self.prompt_embeddings = self.embed_texts(prompts)
        
        # Construction de l'index FAISS si demandé
        if self.use_faiss:
            self._build_faiss_index()
        
        logger.info(f"Index des embeddings construit: {self.prompt_embeddings.shape}")
    
    def _build_faiss_index(self) -> None:
        """
        Construit l'index FAISS pour une recherche plus rapide.
        """
        dimension = self.prompt_embeddings.shape[1]
        
        # Index FAISS pour la recherche par similarité cosinus
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine similarity
        
        # Normalisation des vecteurs pour la similarité cosinus
        faiss.normalize_L2(self.prompt_embeddings)
        
        # Ajout des vecteurs à l'index
        self.faiss_index.add(self.prompt_embeddings.astype('float32'))
        
        logger.info("Index FAISS construit avec succès")
    
    def retrieve_similar_prompts(self, 
                                query_prompt: str, 
                                top_k: int = 5,
                                similarity_threshold: float = 0.5,
                                diversify_results: bool = True) -> List[Dict]:
        """
        Récupère les prompts les plus similaires à la requête avec option de diversification.
        
        Args:
            query_prompt: Le prompt de requête
            top_k: Nombre de résultats à retourner
            similarity_threshold: Seuil de similarité minimum
            diversify_results: Si True, diversifie les résultats pour éviter la répétition
            
        Returns:
            Liste des dictionnaires avec 'prompt', 'response', 'similarity_score'
        """
        if self.prompt_embeddings is None:
            raise ValueError("Index des embeddings non construit. Utilisez build_embeddings_index() d'abord.")
        
        # Embedding de la requête
        query_embedding = self.embed_texts([query_prompt])
        
        # Recherche de similarité avec plus de candidats pour la diversification
        search_k = top_k * 3 if diversify_results else top_k
        
        if self.use_faiss and self.faiss_index is not None:
            similarities, indices = self._search_faiss(query_embedding, search_k)
        else:
            similarities, indices = self._search_cosine(query_embedding, search_k)
        
        # Filtrage par seuil et construction des résultats
        candidates = []
        for sim_score, idx in zip(similarities[0], indices[0]):
            if sim_score >= similarity_threshold:
                candidates.append({
                    'prompt': self.dataset[idx]['prompt'],
                    'response': self.dataset[idx]['response'],
                    'similarity_score': float(sim_score),
                    'index': idx
                })
        
        # Diversification des résultats si demandé
        if diversify_results and len(candidates) > top_k:
            results = self._diversify_results(candidates, top_k, query_prompt)
        else:
            results = candidates[:top_k]
        
        return results
    
    def _diversify_results(self, candidates: List[Dict], top_k: int, query_prompt: str) -> List[Dict]:
        """
        Diversifie les résultats pour éviter la répétition d'exemples similaires.
        
        Args:
            candidates: Liste des candidats triés par similarité
            top_k: Nombre de résultats souhaités
            query_prompt: Le prompt de requête original
            
        Returns:
            Liste diversifiée des résultats
        """
        if len(candidates) <= top_k:
            return candidates
        
        # Sélection du meilleur résultat
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        # Diversification basée sur la similarité entre les candidats
        while len(selected) < top_k and remaining:
            best_candidate = None
            best_diversity_score = -1
            
            for candidate in remaining:
                # Calcul du score de diversité (moins similaire aux déjà sélectionnés)
                diversity_score = self._calculate_diversity_score(candidate, selected, query_prompt)
                
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _calculate_diversity_score(self, candidate: Dict, selected: List[Dict], query_prompt: str) -> float:
        """
        Calcule un score de diversité pour un candidat par rapport aux résultats déjà sélectionnés.
        
        Args:
            candidate: Le candidat à évaluer
            selected: Liste des résultats déjà sélectionnés
            query_prompt: Le prompt de requête original
            
        Returns:
            Score de diversité (plus élevé = plus diversifié)
        """
        if not selected:
            return candidate['similarity_score']
        
        # Score de similarité avec la requête
        query_similarity = candidate['similarity_score']
        
        # Score de diversité par rapport aux résultats sélectionnés
        diversity_penalty = 0
        for selected_item in selected:
            # Calcul de similarité entre les réponses
            similarity = self._calculate_response_similarity(
                candidate['response'], 
                selected_item['response']
            )
            diversity_penalty += similarity
        
        # Score final : similarité avec la requête - pénalité pour la répétition
        diversity_score = query_similarity - (diversity_penalty / len(selected)) * 0.3
        
        return max(0, diversity_score)
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """
        Calcule la similarité entre deux réponses.
        
        Args:
            response1: Première réponse
            response2: Deuxième réponse
            
        Returns:
            Score de similarité entre 0 et 1
        """
        try:
            # Embeddings des réponses
            embeddings = self.embed_texts([response1, response2])
            
            # Similarité cosinus
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"Erreur lors du calcul de similarité des réponses: {e}")
            return 0.0
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recherche utilisant FAISS.
        """
        # Normalisation de la requête
        faiss.normalize_L2(query_embedding)
        
        # Recherche
        similarities, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        return similarities, indices
    
    def _search_cosine(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recherche utilisant la similarité cosinus avec scikit-learn.
        """
        # Calcul de la similarité cosinus
        similarities = cosine_similarity(query_embedding, self.prompt_embeddings)
        
        # Obtention des top-k indices
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        top_similarities = similarities[0][top_indices]
        
        return np.array([top_similarities]), np.array([top_indices])
    
    def retrieve_context(self, user_prompt: str, top_k: int = 5, diversify: bool = True) -> List[str]:
        """
        Fonction principale pour récupérer le contexte RAG avec option de diversification.
        
        Args:
            user_prompt: Le prompt utilisateur
            top_k: Nombre de réponses à récupérer
            diversify: Si True, diversifie les résultats pour éviter la répétition
            
        Returns:
            Liste des réponses les plus pertinentes
        """
        similar_prompts = self.retrieve_similar_prompts(
            user_prompt, 
            top_k=top_k, 
            diversify_results=diversify
        )
        return [item['response'] for item in similar_prompts]
    
    def save_embeddings(self, filepath: str) -> None:
        """
        Sauvegarde les embeddings sur disque.
        
        Args:
            filepath: Chemin de sauvegarde
        """
        if self.prompt_embeddings is not None:
            np.save(filepath, self.prompt_embeddings)
            logger.info(f"Embeddings sauvegardés: {filepath}")
    
    def load_embeddings(self, filepath: str) -> None:
        """
        Charge les embeddings depuis le disque.
        
        Args:
            filepath: Chemin des embeddings sauvegardés
        """
        if os.path.exists(filepath):
            self.prompt_embeddings = np.load(filepath)
            logger.info(f"Embeddings chargés: {filepath}")
        else:
            logger.warning(f"Fichier d'embeddings non trouvé: {filepath}")


def retrieve_context(user_prompt: str, dataset: List[Dict], 
                    model_name: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
                    top_k: int = 5,
                    similarity_threshold: float = 0.5,
                    diversify: bool = True) -> List[str]:
    """
    Fonction utilitaire pour récupérer le contexte RAG avec option de diversification.
    
    Args:
        user_prompt: Le prompt utilisateur
        dataset: Liste de dictionnaires avec 'prompt' et 'response'
        model_name: Nom du modèle d'embeddings
        top_k: Nombre de réponses à récupérer
        similarity_threshold: Seuil de similarité minimum
        diversify: Si True, diversifie les résultats pour éviter la répétition
        
    Returns:
        Liste des réponses les plus pertinentes
    """
    # Initialisation du système RAG
    rag_system = ArabicMatryoshkaRAG(model_name=model_name)
    
    # Chargement du dataset
    rag_system.load_dataset_from_list(dataset)
    
    # Construction de l'index
    rag_system.build_embeddings_index()
    
    # Récupération du contexte avec diversification
    return rag_system.retrieve_context(user_prompt, top_k, diversify=diversify)


# Exemple d'utilisation
if __name__ == "__main__":
    # Test du système RAG
    test_dataset = [
        {
            "prompt": "أنت مذيع نشرة جوية محترف، تتحدث باللغة العربية الفصحى...",
            "response": "أسعد الله أوقاتكم بكل خير، إليكم النشرة الجوية..."
        },
        {
            "prompt": "توليد نشرة جوية بناءً على بيانات الطقس...",
            "response": "أهلاً ومرحباً بكم في نشرتنا الجوية..."
        }
    ]
    
    # Test de la fonction utilitaire
    user_query = "أنت مذيع نشرة جوية محترف، توليد نشرة جوية"
    context = retrieve_context(user_query, test_dataset, top_k=3)
    
    print("Contexte récupéré:")
    for i, response in enumerate(context, 1):
        print(f"{i}. {response[:100]}...") 