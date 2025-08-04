#!/usr/bin/env python3
"""
Test du système de cache intelligent pour les prompts météorologiques.
"""

import json
import time
from cache_system import WeatherCache, get_cached_response, add_to_cache

def test_basic_cache_operations():
    """Test des opérations de base du cache."""
    print("🧪 Test des opérations de base du cache...")
    
    # Initialisation du cache
    cache = WeatherCache("test_cache.json")
    
    # Test d'ajout d'entrées
    test_prompts = [
        "أنت مذيع نشرة جوية محترف. Casablanca 22 Ciel_Clair",
        "أنت مذيع نشرة جوية محترف. Rabat 21 Peu_Nuageux",
        "أنت مذيع نشرة جوية محترف. Marrakech 28 Ciel_Clair"
    ]
    
    test_responses = [
        "أسعد الله أوقاتكم بكل خير، إليكم النشرة الجوية المتوقعة ليوم غد في مختلف مناطق المملكة.",
        "أهلاً ومرحباً بكم مشاهدينا الكرام في نشرتنا الجوية المسائية.",
        "نرحب بكم في نشرتنا الجوية لهذا اليوم، حيث نتابع معكم آخر مستجدات الحالة الجوية."
    ]
    
    # Ajout des entrées de test
    for prompt, response in zip(test_prompts, test_responses):
        cache.add_to_cache(prompt, response, {"test": True})
        print(f"✅ Ajouté au cache: {prompt[:50]}...")
    
    # Test de récupération exacte
    print("\n🔍 Test de récupération exacte...")
    for prompt in test_prompts:
        cached_response = cache.get_from_cache(prompt)
        if cached_response:
            print(f"✅ Cache hit exact: {cached_response[:50]}...")
        else:
            print(f"❌ Cache miss: {prompt[:50]}...")
    
    # Test de récupération par similarité
    print("\n🔍 Test de récupération par similarité...")
    similar_queries = [
        "أنت مذيع نشرة جوية محترف. Casablanca 22 Ciel_Clair",  # Exact
        "أنت مذيع نشرة جوية محترف. Casablanca 22",  # Similaire
        "أنت مذيع نشرة جوية محترف. Rabat 21",  # Similaire
        "أنت مذيع نشرة جوية محترف. Tanger 19 Ciel_Clair"  # Différent
    ]
    
    for query in similar_queries:
        cached_response = cache.get_from_cache(query)
        if cached_response:
            print(f"✅ Cache hit: {cached_response[:50]}...")
        else:
            print(f"❌ Cache miss: {query[:50]}...")

def test_cache_performance():
    """Test de performance du cache."""
    print("\n🧪 Test de performance du cache...")
    
    cache = WeatherCache("test_cache.json")
    
    # Test de temps de recherche
    test_prompt = "أنت مذيع نشرة جوية محترف. Casablanca 22 Ciel_Clair"
    
    # Recherche dans le cache
    start_time = time.time()
    cached_response = cache.get_from_cache(test_prompt)
    cache_time = time.time() - start_time
    
    print(f"⏱️ Temps de recherche dans le cache: {cache_time:.4f} secondes")
    
    if cached_response:
        print(f"✅ Réponse trouvée: {cached_response[:50]}...")
    else:
        print("❌ Aucune réponse trouvée")
    
    # Test de temps d'ajout
    new_prompt = "أنت مذيع نشرة جوية محترف. Fes 24 Peu_Nuageux"
    new_response = "أهلاً ومرحباً بكم في نشرتنا الجوية لهذا اليوم."
    
    start_time = time.time()
    cache.add_to_cache(new_prompt, new_response)
    add_time = time.time() - start_time
    
    print(f"⏱️ Temps d'ajout au cache: {add_time:.4f} secondes")

def test_cache_statistics():
    """Test des statistiques du cache."""
    print("\n🧪 Test des statistiques du cache...")
    
    cache = WeatherCache("test_cache.json")
    
    # Ajout de quelques entrées supplémentaires
    additional_prompts = [
        "أنت مذيع نشرة جوية محترف. Tanger 19 Ciel_Clair",
        "أنت مذيع نشرة جوية محترف. Agadir 25 Ciel_Clair",
        "أنت مذيع نشرة جوية محترف. Oujda 26 Peu_Nuageux"
    ]
    
    additional_responses = [
        "أسعد الله أوقاتكم بكل خير، إليكم النشرة الجوية...",
        "أهلاً ومرحباً بكم في نشرتنا الجوية...",
        "نرحب بكم في نشرتنا الجوية لهذا اليوم..."
    ]
    
    for prompt, response in zip(additional_prompts, additional_responses):
        cache.add_to_cache(prompt, response)
    
    # Affichage des statistiques
    stats = cache.get_cache_stats()
    print("📊 Statistiques du cache:")
    print(f"  - Nombre total d'entrées: {stats['total_entries']}")
    print(f"  - Taille totale: {stats['total_size_bytes']} bytes")
    print(f"  - Âge moyen: {stats['average_age_hours']:.2f} heures")
    print(f"  - Seuil de similarité: {stats['similarity_threshold']}")

def test_cache_search():
    """Test de recherche dans le cache."""
    print("\n🧪 Test de recherche dans le cache...")
    
    cache = WeatherCache("test_cache.json")
    
    # Recherche par similarité
    search_query = "أنت مذيع نشرة جوية محترف"
    results = cache.search_cache(search_query, limit=3)
    
    print(f"🔍 Résultats pour '{search_query}':")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Similarité: {result['similarity']:.3f}")
        print(f"     Prompt: {result['prompt'][:50]}...")
        print(f"     Réponse: {result['response'][:50]}...")
        print()

def test_cache_cleanup():
    """Test du nettoyage du cache."""
    print("\n🧪 Test du nettoyage du cache...")
    
    cache = WeatherCache("test_cache.json")
    
    # Affichage des statistiques avant nettoyage
    stats_before = cache.get_cache_stats()
    print(f"📊 Avant nettoyage: {stats_before['total_entries']} entrées")
    
    # Nettoyage des entrées anciennes (plus de 1 jour)
    removed = cache.clear_cache(older_than_days=1)
    print(f"🗑️ Entrées supprimées: {removed}")
    
    # Affichage des statistiques après nettoyage
    stats_after = cache.get_cache_stats()
    print(f"📊 Après nettoyage: {stats_after['total_entries']} entrées")

def test_cache_integration():
    """Test d'intégration avec les fonctions utilitaires."""
    print("\n🧪 Test d'intégration avec les fonctions utilitaires...")
    
    # Test avec les fonctions utilitaires
    test_prompt = "أنت مذيع نشرة جوية محترف. Test Integration"
    test_response = "Ceci est un test d'intégration du système de cache."
    
    # Ajout avec la fonction utilitaire
    add_to_cache(test_prompt, test_response, "test_cache.json")
    print("✅ Ajouté avec la fonction utilitaire")
    
    # Récupération avec la fonction utilitaire
    cached_response = get_cached_response(test_prompt, "test_cache.json")
    if cached_response:
        print(f"✅ Récupéré avec la fonction utilitaire: {cached_response[:50]}...")
    else:
        print("❌ Échec de récupération avec la fonction utilitaire")

def main():
    """Fonction principale de test."""
    print("🚀 Test du système de cache intelligent")
    print("=" * 60)
    
    # Tests
    test_basic_cache_operations()
    test_cache_performance()
    test_cache_statistics()
    test_cache_search()
    test_cache_cleanup()
    test_cache_integration()
    
    print("\n" + "=" * 60)
    print("✅ Tous les tests terminés!")
    print("\n💡 Avantages du système de cache:")
    print("- Réponses instantanées pour les prompts similaires")
    print("- Réduction significative des appels à Ollama")
    print("- Évite les timeouts pour les données répétitives")
    print("- Améliore l'expérience utilisateur")

if __name__ == "__main__":
    main() 