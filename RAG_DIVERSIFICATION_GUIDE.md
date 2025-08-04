# Guide de Diversification du Système RAG

## 🚀 Vue d'ensemble

Le système RAG a été amélioré avec des fonctionnalités de diversification pour éviter la répétition des mêmes exemples et améliorer la variété du contexte fourni.

## 🔍 Problème Identifié

### Contexte RAG Répétitif
Le système RAG retournait toujours les mêmes exemples :
```
Contexte RAG:
- أسعد الله أوقاتكم بكل خير، وأهلاً ومرحباً بكم مشاهدينا الكرام في هذه النشرة الجوية ليوم الجمعة 28 ما ا...
- **النشرة الجوية ليوم الجمعة 28 مارس 2025م** أسعد الله أوقاتكم بكل خير، إليكم أحوال الطقس المتوقعة لي...
```

### Causes
1. **Dataset limité** : Seulement 11 exemples dans `dataset.json`
2. **Dataset homogène** : Tous les exemples datent du 28 mars 2025
3. **Recherche basique** : Pas de diversification des résultats

## ✨ Solutions Implémentées

### 1. Système de Diversification

#### Algorithme de Diversification
```python
def _diversify_results(self, candidates: List[Dict], top_k: int, query_prompt: str) -> List[Dict]:
    """
    Diversifie les résultats pour éviter la répétition d'exemples similaires.
    """
    # Sélection du meilleur résultat
    selected = [candidates[0]]
    remaining = candidates[1:]
    
    # Diversification basée sur la similarité entre les candidats
    while len(selected) < top_k and remaining:
        best_candidate = None
        best_diversity_score = -1
        
        for candidate in remaining:
            diversity_score = self._calculate_diversity_score(candidate, selected, query_prompt)
            if diversity_score > best_diversity_score:
                best_diversity_score = diversity_score
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
    
    return selected
```

#### Calcul du Score de Diversité
```python
def _calculate_diversity_score(self, candidate: Dict, selected: List[Dict], query_prompt: str) -> float:
    """
    Calcule un score de diversité pour un candidat.
    """
    # Score de similarité avec la requête
    query_similarity = candidate['similarity_score']
    
    # Pénalité pour la répétition
    diversity_penalty = 0
    for selected_item in selected:
        similarity = self._calculate_response_similarity(
            candidate['response'], 
            selected_item['response']
        )
        diversity_penalty += similarity
    
    # Score final : similarité - pénalité
    diversity_score = query_similarity - (diversity_penalty / len(selected)) * 0.3
    
    return max(0, diversity_score)
```

### 2. Similarité entre Réponses

#### Calcul de Similarité Sémantique
```python
def _calculate_response_similarity(self, response1: str, response2: str) -> float:
    """
    Calcule la similarité entre deux réponses.
    """
    # Embeddings des réponses
    embeddings = self.embed_texts([response1, response2])
    
    # Similarité cosinus
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    
    return float(similarity)
```

### 3. Recherche Élargie

#### Plus de Candidats pour la Diversification
```python
# Recherche de similarité avec plus de candidats pour la diversification
search_k = top_k * 3 if diversify_results else top_k

if self.use_faiss and self.faiss_index is not None:
    similarities, indices = self._search_faiss(query_embedding, search_k)
else:
    similarities, indices = self._search_cosine(query_embedding, search_k)
```

## 🔧 Utilisation

### 1. Dans le Système RAG de Base

```python
# Sans diversification (ancien comportement)
results = rag_system.retrieve_similar_prompts(
    query_prompt, 
    top_k=3, 
    diversify_results=False
)

# Avec diversification (nouveau comportement)
results = rag_system.retrieve_similar_prompts(
    query_prompt, 
    top_k=3, 
    diversify_results=True  # Par défaut
)
```

### 2. Dans le Générateur RAG Intégré

```python
# Sans diversification
enhanced_prompt = rag_generator.enhance_prompt_with_rag(
    base_prompt, 
    weather_data, 
    top_k=2, 
    diversify=False
)

# Avec diversification
enhanced_prompt = rag_generator.enhance_prompt_with_rag(
    base_prompt, 
    weather_data, 
    top_k=2, 
    diversify=True  # Par défaut
)
```

### 3. Fonction Utilitaire

```python
# Avec diversification
rag_context = retrieve_context(
    user_prompt, 
    dataset, 
    top_k=3, 
    diversify=True  # Par défaut
)
```

## 📊 Avantages

### 1. Variété du Contexte
- **Avant** : Mêmes exemples répétés
- **Après** : Exemples diversifiés et variés

### 2. Optimisation du Dataset
- Utilisation plus efficace du dataset limité
- Évite la surutilisation des mêmes exemples

### 3. Qualité des Réponses
- Contexte RAG plus riche et diversifié
- Réponses plus dynamiques et variées

### 4. Performance
- Recherche élargie (3x plus de candidats)
- Sélection intelligente basée sur la diversité

## 🧪 Tests

### Script de Test
```bash
python test_rag_diversification.py
```

### Métriques Testées
- Similarité entre réponses
- Score de diversité
- Comparaison avec/sans diversification

## 🔄 Configuration

### Paramètres de Diversification

#### Seuil de Diversité
```python
# Dans _calculate_diversity_score
diversity_score = query_similarity - (diversity_penalty / len(selected)) * 0.3
#                                                                        ^
#                                                              Facteur de pénalité
```

#### Nombre de Candidats
```python
# Recherche élargie
search_k = top_k * 3 if diversify_results else top_k
#                    ^
#            Facteur d'élargissement
```

### Personnalisation

#### Ajustement du Facteur de Pénalité
```python
# Plus de diversification
diversity_score = query_similarity - (diversity_penalty / len(selected)) * 0.5

# Moins de diversification
diversity_score = query_similarity - (diversity_penalty / len(selected)) * 0.1
```

#### Ajustement du Facteur d'Élargissement
```python
# Plus de candidats
search_k = top_k * 5 if diversify_results else top_k

# Moins de candidats
search_k = top_k * 2 if diversify_results else top_k
```

## 📈 Résultats Attendus

### Avant la Diversification
```
Contexte RAG:
- أسعد الله أوقاتكم بكل خير، وأهلاً ومرحباً بكم مشاهدينا الكرام...
- **النشرة الجوية ليوم الجمعة 28 مارس 2025م** أسعد الله أوقاتكم بكل خير...
```

### Après la Diversification
```
Contexte RAG:
- أسعد الله أوقاتكم بكل خير، وأهلاً ومرحباً بكم مشاهدينا الكرام...
- أهلاً ومرحباً بكم مشاهدينا الكرام في نشرتنا الجوية لهذا اليوم...
```

## 🚀 Prochaines Étapes

### 1. Enrichissement du Dataset
- Ajouter plus d'exemples variés
- Inclure différents styles de présentation
- Diversifier les dates et contextes

### 2. Optimisations Avancées
- Clustering des exemples similaires
- Sélection basée sur la qualité
- Apprentissage des préférences utilisateur

### 3. Monitoring
- Métriques de diversité en temps réel
- Feedback utilisateur sur la qualité
- Ajustement automatique des paramètres

---

**Note** : La diversification est activée par défaut dans toutes les fonctions pour améliorer automatiquement la variété du contexte RAG. 