"""
Script de test pour le système RAG avec embeddings Arabic Matryoshka.
"""

import json
import time
from RAG import ArabicMatryoshkaRAG, retrieve_context
from rag_integration import RAGWeatherGenerator, enhance_prompt_with_rag

def test_basic_rag():
    """Test basique du système RAG."""
    print("🧪 Test basique du système RAG...")
    
    # Dataset de test
    test_dataset = [
        {
            "prompt": "أنت مذيع نشرة جوية محترف، تتحدث باللغة العربية الفصحى بصيغة رسمية ومناسبة للبث التلفزيوني. مهمتك هي توليد نشرة جوية طبيعية وسلسة بناءً على بيانات الطقس التي ستزود بها.",
            "response": "أسعد الله أوقاتكم بكل خير، إليكم النشرة الجوية المتوقعة ليوم غد في مختلف مناطق المملكة. تشير التوقعات إلى أجواء معتدلة بشكل عام مع تفاوت بسيط في درجات الحرارة."
        },
        {
            "prompt": "توليد نشرة جوية بناءً على بيانات الطقس المتوفرة، استخدام اللغة العربية الفصحى للبث التلفزيوني.",
            "response": "أهلاً ومرحباً بكم في نشرتنا الجوية لهذا اليوم. من المتوقع أن تكون الأجواء صافية إلى قليلة السحب في مختلف المناطق."
        },
        {
            "prompt": "مذيع نشرة جوية محترف يقدم توقعات الطقس باللغة العربية للجمهور التلفزيوني.",
            "response": "نرحب بكم في نشرتنا الجوية. تشير التوقعات إلى سماء غائمة جزئياً مع احتمال تساقط بعض الأمطار المحلية."
        }
    ]
    
    # Test de la fonction utilitaire
    user_query = "أنت مذيع نشرة جوية محترف، توليد نشرة جوية"
    print(f"🔍 Requête: {user_query}")
    
    try:
        context = retrieve_context(user_query, test_dataset, top_k=2)
        print(f"✅ Contexte récupéré ({len(context)} exemples):")
        for i, response in enumerate(context, 1):
            print(f"  {i}. {response[:100]}...")
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")

def test_rag_with_real_dataset():
    """Test du RAG avec le vrai dataset."""
    print("\n🧪 Test du RAG avec le dataset réel...")
    
    try:
        # Chargement du dataset
        with open("dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        print(f"📊 Dataset chargé: {len(dataset)} exemples")
        
        # Test avec différents prompts
        test_queries = [
            "أنت مذيع نشرة جوية محترف، توليد نشرة جوية",
            "توقعات الطقس للمغرب",
            "نشرة جوية باللغة العربية"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Test avec la requête: {query}")
            try:
                context = retrieve_context(query, dataset, top_k=3)
                print(f"✅ Contexte récupéré ({len(context)} exemples):")
                for i, response in enumerate(context, 1):
                    print(f"  {i}. {response[:150]}...")
            except Exception as e:
                print(f"❌ Erreur: {e}")
                
    except Exception as e:
        print(f"❌ Erreur lors du chargement du dataset: {e}")

def test_rag_integration():
    """Test de l'intégration RAG."""
    print("\n🧪 Test de l'intégration RAG...")
    
    try:
        # Test avec des données météorologiques simulées
        weather_data = """
        Ville Tmax Phenomenes
        Casablanca 22 Ciel_Clair
        Rabat 21 Peu_Nuageux
        Marrakech 28 Ciel_Clair
        """
        
        base_prompt = "أنت مذيع نشرة جوية محترف، تتحدث باللغة العربية الفصحى"
        
        # Chargement du dataset
        with open("dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        # Test de l'amélioration du prompt
        enhanced_prompt = enhance_prompt_with_rag(base_prompt, weather_data, dataset, top_k=2)
        
        print("✅ Prompt amélioré avec RAG:")
        print(enhanced_prompt[:500] + "...")
        
    except Exception as e:
        print(f"❌ Erreur lors du test d'intégration: {e}")

def test_rag_performance():
    """Test de performance du système RAG."""
    print("\n🧪 Test de performance du RAG...")
    
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
        
        # Test de recherche
        query = "أنت مذيع نشرة جوية محترف، توليد نشرة جوية"
        start_time = time.time()
        
        results = rag_system.retrieve_similar_prompts(query, top_k=5)
        
        search_time = time.time() - start_time
        print(f"⏱️ Temps de recherche: {search_time:.3f} secondes")
        print(f"📊 Résultats trouvés: {len(results)}")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['similarity_score']:.3f}")
            print(f"     Réponse: {result['response'][:100]}...")
        
    except Exception as e:
        print(f"❌ Erreur lors du test de performance: {e}")

def test_rag_with_faiss():
    """Test du RAG avec FAISS."""
    print("\n🧪 Test du RAG avec FAISS...")
    
    try:
        # Test avec et sans FAISS
        test_dataset = [
            {
                "prompt": "أنت مذيع نشرة جوية محترف",
                "response": "أسعد الله أوقاتكم بكل خير، إليكم النشرة الجوية..."
            },
            {
                "prompt": "توليد نشرة جوية بناءً على بيانات الطقس",
                "response": "أهلاً ومرحباً بكم في نشرتنا الجوية..."
            }
        ]
        
        # Test sans FAISS
        print("🔍 Test sans FAISS...")
        rag_no_faiss = ArabicMatryoshkaRAG(use_faiss=False)
        rag_no_faiss.load_dataset_from_list(test_dataset)
        rag_no_faiss.build_embeddings_index()
        
        results_no_faiss = rag_no_faiss.retrieve_similar_prompts("مذيع نشرة جوية", top_k=2)
        print(f"✅ Résultats sans FAISS: {len(results_no_faiss)}")
        
        # Test avec FAISS
        print("🔍 Test avec FAISS...")
        rag_with_faiss = ArabicMatryoshkaRAG(use_faiss=True)
        rag_with_faiss.load_dataset_from_list(test_dataset)
        rag_with_faiss.build_embeddings_index()
        
        results_with_faiss = rag_with_faiss.retrieve_similar_prompts("مذيع نشرة جوية", top_k=2)
        print(f"✅ Résultats avec FAISS: {len(results_with_faiss)}")
        
    except Exception as e:
        print(f"❌ Erreur lors du test FAISS: {e}")

if __name__ == "__main__":
    print("🚀 Démarrage des tests du système RAG...")
    
    # Tests
    test_basic_rag()
    test_rag_with_real_dataset()
    test_rag_integration()
    test_rag_performance()
    test_rag_with_faiss()
    
    print("\n✅ Tous les tests terminés!") 