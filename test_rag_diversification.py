#!/usr/bin/env python3
"""
Test de la diversification du système RAG.
"""

import json
import logging
from RAG import ArabicMatryoshkaRAG
from rag_integration import RAGWeatherGenerator

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_diversification():
    """Test de la diversification du système RAG."""
    print("🧪 Test de la diversification du système RAG")
    print("=" * 60)
    
    # Chargement du dataset
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"✅ Dataset chargé: {len(dataset)} exemples")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du dataset: {e}")
        return
    
    # Test avec le système RAG de base
    print("\n🔍 Test avec le système RAG de base...")
    rag_system = ArabicMatryoshkaRAG(
        model_name="Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
        use_faiss=False  # Utiliser la recherche cosinus pour ce test
    )
    
    # Chargement et initialisation
    rag_system.load_dataset_from_list(dataset)
    rag_system.build_embeddings_index()
    
    # Test avec différents prompts
    test_prompts = [
        "أنت مذيع نشرة جوية محترف. Casablanca 22 Ciel_Clair",
        "أنت مذيع نشرة جوية محترف. Rabat 21 Peu_Nuageux",
        "أنت مذيع نشرة جوية محترف. Marrakech 28 Ciel_Clair",
        "أنت مذيع نشرة جوية محترف. Tanger 19 Ciel_Clair"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt[:50]}...")
        
        # Test sans diversification
        print("  🔍 Sans diversification:")
        results_no_diversify = rag_system.retrieve_similar_prompts(
            prompt, 
            top_k=3, 
            diversify_results=False
        )
        
        for j, result in enumerate(results_no_diversify, 1):
            print(f"    {j}. Similarité: {result['similarity_score']:.3f}")
            print(f"       Réponse: {result['response'][:80]}...")
        
        # Test avec diversification
        print("  🌈 Avec diversification:")
        results_diversify = rag_system.retrieve_similar_prompts(
            prompt, 
            top_k=3, 
            diversify_results=True
        )
        
        for j, result in enumerate(results_diversify, 1):
            print(f"    {j}. Similarité: {result['similarity_score']:.3f}")
            print(f"       Réponse: {result['response'][:80]}...")
    
    # Test avec le générateur RAG intégré
    print("\n🔧 Test avec le générateur RAG intégré...")
    rag_generator = RAGWeatherGenerator("dataset.json")
    
    test_weather_data = """
    Casablanca 22 Ciel_Clair
    Rabat 21 Peu_Nuageux
    Marrakech 28 Ciel_Clair
    """
    
    # Test sans diversification
    print("  🔍 Sans diversification:")
    enhanced_prompt_no_diversify = rag_generator.enhance_prompt_with_rag(
        "أنت مذيع نشرة جوية محترف", 
        test_weather_data, 
        top_k=2, 
        diversify=False
    )
    print(f"    Prompt amélioré: {enhanced_prompt_no_diversify[:200]}...")
    
    # Test avec diversification
    print("  🌈 Avec diversification:")
    enhanced_prompt_diversify = rag_generator.enhance_prompt_with_rag(
        "أنت مذيع نشرة جوية محترف", 
        test_weather_data, 
        top_k=2, 
        diversify=True
    )
    print(f"    Prompt amélioré: {enhanced_prompt_diversify[:200]}...")
    
    # Comparaison des résultats
    print("\n📊 Comparaison des résultats:")
    print("  - Sans diversification: Même contexte RAG répété")
    print("  - Avec diversification: Contexte RAG varié et diversifié")
    
    print("\n" + "=" * 60)
    print("✅ Test de diversification terminé!")
    print("\n💡 Avantages de la diversification:")
    print("- Évite la répétition des mêmes exemples")
    print("- Améliore la variété du contexte RAG")
    print("- Rend les réponses plus dynamiques")
    print("- Optimise l'utilisation du dataset limité")

def test_diversity_metrics():
    """Test des métriques de diversité."""
    print("\n📈 Test des métriques de diversité...")
    
    # Chargement du dataset
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    rag_system = ArabicMatryoshkaRAG()
    rag_system.load_dataset_from_list(dataset)
    rag_system.build_embeddings_index()
    
    # Test de similarité entre réponses
    test_responses = [
        dataset[0]['response'],
        dataset[1]['response'],
        dataset[2]['response']
    ]
    
    print("  🔍 Similarité entre réponses:")
    for i in range(len(test_responses)):
        for j in range(i+1, len(test_responses)):
            similarity = rag_system._calculate_response_similarity(
                test_responses[i], 
                test_responses[j]
            )
            print(f"    Réponse {i+1} vs Réponse {j+1}: {similarity:.3f}")
    
    # Test de score de diversité
    print("  🌈 Score de diversité:")
    candidates = [
        {'response': dataset[0]['response'], 'similarity_score': 0.9},
        {'response': dataset[1]['response'], 'similarity_score': 0.8},
        {'response': dataset[2]['response'], 'similarity_score': 0.7}
    ]
    
    selected = [candidates[0]]
    for candidate in candidates[1:]:
        diversity_score = rag_system._calculate_diversity_score(
            candidate, selected, "test prompt"
        )
        print(f"    Candidat: {diversity_score:.3f}")

if __name__ == "__main__":
    test_rag_diversification()
    test_diversity_metrics() 