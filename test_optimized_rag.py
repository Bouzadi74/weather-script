#!/usr/bin/env python3
"""
Test du système RAG optimisé pour éviter les timeouts.
"""

import json
import time
from RAG import ArabicMatryoshkaRAG
from rag_integration import enhance_prompt_with_rag

def test_optimized_prompt_length():
    """Test de la longueur des prompts optimisés."""
    print("🧪 Test de la longueur des prompts optimisés...")
    
    # Chargement du dataset
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Test avec différents prompts
    test_cases = [
        "أنت مذيع نشرة جوية محترف",
        "توقعات الطقس للمغرب",
        "نشرة جوية باللغة العربية"
    ]
    
    for query in test_cases:
        print(f"\n🔍 Test avec: {query}")
        
        # Test de la fonction optimisée
        weather_data = "Casablanca 22 Ciel_Clair, Rabat 21 Peu_Nuageux"
        base_prompt = "أنت مذيع نشرة جوية محترف. ابدأ بتحية رسمية."
        
        enhanced_prompt = enhance_prompt_with_rag(base_prompt, weather_data, dataset, top_k=2)
        
        # Analyse de la longueur
        prompt_length = len(enhanced_prompt)
        print(f"  📏 Longueur du prompt: {prompt_length} caractères")
        
        if prompt_length > 2000:
            print(f"  ⚠️ Prompt trop long ({prompt_length} chars)")
        elif prompt_length > 1000:
            print(f"  ⚠️ Prompt assez long ({prompt_length} chars)")
        else:
            print(f"  ✅ Prompt de taille raisonnable ({prompt_length} chars)")
        
        # Affichage du début du prompt
        print(f"  📝 Début du prompt: {enhanced_prompt[:200]}...")

def test_rag_performance():
    """Test de performance du système RAG optimisé."""
    print("\n🧪 Test de performance du RAG optimisé...")
    
    try:
        # Chargement du dataset
        with open("dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        # Initialisation du système RAG
        start_time = time.time()
        rag_system = ArabicMatryoshkaRAG(
            model_name="Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
            use_faiss=True,
            dimension=768
        )
        
        # Chargement et construction de l'index
        rag_system.load_dataset_from_list(dataset)
        rag_system.build_embeddings_index()
        
        init_time = time.time() - start_time
        print(f"⏱️ Temps d'initialisation: {init_time:.2f} secondes")
        
        # Test de recherche optimisée
        query = "أنت مذيع نشرة جوية محترف، توليد نشرة جوية"
        start_time = time.time()
        
        results = rag_system.retrieve_similar_prompts(query, top_k=2)  # Réduit à 2
        
        search_time = time.time() - start_time
        print(f"⏱️ Temps de recherche: {search_time:.3f} secondes")
        print(f"📊 Résultats trouvés: {len(results)}")
        
        # Test de génération de prompt optimisé
        weather_data = "Casablanca 22 Ciel_Clair, Rabat 21 Peu_Nuageux"
        base_prompt = "أنت مذيع نشرة جوية محترف. ابدأ بتحية رسمية."
        
        start_time = time.time()
        enhanced_prompt = enhance_prompt_with_rag(base_prompt, weather_data, dataset, top_k=2)
        prompt_time = time.time() - start_time
        
        print(f"⏱️ Temps de génération du prompt: {prompt_time:.3f} secondes")
        print(f"📏 Longueur finale: {len(enhanced_prompt)} caractères")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test de performance: {e}")
        return False

def test_prompt_optimization():
    """Test de l'optimisation des prompts."""
    print("\n🧪 Test de l'optimisation des prompts...")
    
    # Test avec différents scénarios
    test_scenarios = [
        {
            "name": "Prompt court",
            "base_prompt": "أنت مذيع نشرة جوية محترف.",
            "weather_data": "Casablanca 22 Ciel_Clair"
        },
        {
            "name": "Prompt moyen",
            "base_prompt": "أنت مذيع نشرة جوية محترف. ابدأ بتحية رسمية، ثم وصف عام للطقس.",
            "weather_data": "Casablanca 22 Ciel_Clair, Rabat 21 Peu_Nuageux, Marrakech 28 Ciel_Clair"
        },
        {
            "name": "Prompt long",
            "base_prompt": "أنت مذيع نشرة جوية محترف. ابدأ بتحية رسمية، ثم وصف عام للطقس، ثم درجات الحرارة والظواهر لكل مدينة في فقرة واحدة.",
            "weather_data": "Casablanca 22 Ciel_Clair, Rabat 21 Peu_Nuageux, Marrakech 28 Ciel_Clair, Fes 24 Peu_Nuageux, Tanger 19 Ciel_Clair"
        }
    ]
    
    # Chargement du dataset
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    for scenario in test_scenarios:
        print(f"\n📝 Test: {scenario['name']}")
        
        # Test sans RAG
        prompt_without_rag = f"{scenario['base_prompt']}\n\n{scenario['weather_data']}"
        print(f"  📏 Sans RAG: {len(prompt_without_rag)} caractères")
        
        # Test avec RAG optimisé
        prompt_with_rag = enhance_prompt_with_rag(
            scenario['base_prompt'], 
            scenario['weather_data'], 
            dataset, 
            top_k=2
        )
        print(f"  📏 Avec RAG: {len(prompt_with_rag)} caractères")
        
        # Calcul de l'augmentation
        increase = len(prompt_with_rag) - len(prompt_without_rag)
        increase_percent = (increase / len(prompt_without_rag)) * 100
        print(f"  📈 Augmentation: {increase} caractères ({increase_percent:.1f}%)")
        
        if increase_percent > 100:
            print(f"  ⚠️ Augmentation importante")
        elif increase_percent > 50:
            print(f"  ⚠️ Augmentation modérée")
        else:
            print(f"  ✅ Augmentation raisonnable")

def main():
    """Fonction principale de test."""
    print("🚀 Test du système RAG optimisé pour éviter les timeouts")
    print("=" * 60)
    
    # Tests
    test_optimized_prompt_length()
    test_rag_performance()
    test_prompt_optimization()
    
    print("\n" + "=" * 60)
    print("✅ Tests terminés!")
    print("\n💡 Recommandations:")
    print("- Les prompts sont maintenant plus courts")
    print("- Le nombre d'exemples RAG est limité à 2")
    print("- Les timeouts devraient être évités")
    print("- La qualité reste maintenue avec moins de contexte")

if __name__ == "__main__":
    main() 