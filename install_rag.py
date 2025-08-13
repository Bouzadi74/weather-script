#!/usr/bin/env python3
"""
Script d'installation automatique pour le système RAG météorologique.
Installe les dépendances, vérifie la configuration et teste l'installation.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def check_python_version():
    """
    Vérifie que la version de Python est compatible (3.8+).
    
    Returns:
        bool: True si la version est compatible
    """
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis. Version actuelle:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} détecté")
    return True

def install_dependencies():
    """
    Installe les dépendances Python nécessaires pour le système RAG.
    
    Returns:
        bool: True si toutes les dépendances sont installées
    """
    print("📦 Installation des dépendances...")
    
    # Liste des dépendances requises
    dependencies = [
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"  📥 Installation de {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"  ✅ {dep} installé")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Erreur lors de l'installation de {dep}: {e}")
            return False
    
    return True

def check_dataset():
    """
    Vérifie la présence et la validité du dataset de scripts météorologiques.
    
    Returns:
        bool: True si le dataset est valide
    """
    print("📊 Vérification du dataset...")
    
    dataset_path = "dataset.json"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if not isinstance(dataset, list):
            print("❌ Le dataset doit être une liste")
            return False
        
        if len(dataset) == 0:
            print("❌ Le dataset est vide")
            return False
        
        # Vérification de la structure des données
        for i, item in enumerate(dataset):
            if not isinstance(item, dict):
                print(f"❌ Item {i} n'est pas un dictionnaire")
                return False
            
            if 'prompt' not in item or 'response' not in item:
                print(f"❌ Item {i} manque 'prompt' ou 'response'")
                return False
        
        print(f"✅ Dataset valide: {len(dataset)} exemples")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON dans le dataset: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du dataset: {e}")
        return False

def test_imports():
    """
    Teste l'importation des modules requis pour le système RAG.
    
    Returns:
        bool: True si tous les modules peuvent être importés
    """
    print("🧪 Test des imports...")
    
    modules_to_test = [
        "sentence_transformers",
        "faiss",
        "torch",
        "transformers",
        "numpy",
        "sklearn"
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module} importé avec succès")
        except ImportError as e:
            print(f"  ❌ Erreur d'import de {module}: {e}")
            return False
    
    return True

def test_rag_system():
    """Test basique du système RAG."""
    print("🧪 Test du système RAG...")
    
    try:
        from RAG import ArabicMatryoshkaRAG
        
        # Test avec un petit dataset
        test_dataset = [
            {
                "prompt": "أنت مذيع نشرة جوية محترف",
                "response": "أسعد الله أوقاتكم بكل خير، إليكم النشرة الجوية..."
            }
        ]
        
        # Initialisation du système RAG
        rag_system = ArabicMatryoshkaRAG(
            model_name="Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
            use_faiss=False,  # Test sans FAISS d'abord
            dimension=768
        )
        
        # Chargement du dataset
        rag_system.load_dataset_from_list(test_dataset)
        
        # Construction de l'index
        rag_system.build_embeddings_index()
        
        # Test de recherche
        results = rag_system.retrieve_context("مذيع نشرة جوية", top_k=1)
        
        if len(results) > 0:
            print("  ✅ Système RAG fonctionne correctement")
            return True
        else:
            print("  ❌ Aucun résultat trouvé")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur lors du test RAG: {e}")
        return False

def create_sample_dataset():
    """Crée un dataset d'exemple si aucun n'existe."""
    print("📝 Création d'un dataset d'exemple...")
    
    sample_dataset = [
        {
            "prompt": "أنت مذيع نشرة جوية محترف، تتحدث باللغة العربية الفصحى بصيغة رسمية ومناسبة للبث التلفزيوني. مهمتك هي توليد نشرة جوية طبيعية وسلسة بناءً على بيانات الطقس التي ستزود بها.",
            "response": "أسعد الله أوقاتكم بكل خير، إليكم النشرة الجوية المتوقعة ليوم غد في مختلف مناطق المملكة. تشير التوقعات إلى أجواء معتدle بشكل عام مع تفاوت بسيط في درجات الحرارة."
        },
        {
            "prompt": "توليد نشرة جوية بناءً على بيانات الطقس المتوفرة، استخدام اللغة العربية الفصحى للبث التلفزيوني.",
            "response": "أهلاً ومرحباً بكم في نشرتنا الجوية لهذا اليوم. من المتوقع أن تكون الأجواء صافية إلى قليلة السحب في مختلف المناطق."
        }
    ]
    
    try:
        with open("dataset.json", "w", encoding="utf-8") as f:
            json.dump(sample_dataset, f, ensure_ascii=False, indent=2)
        print("✅ Dataset d'exemple créé: dataset.json")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la création du dataset: {e}")
        return False

def main():
    """Fonction principale d'installation."""
    print("🚀 Installation du système RAG avec embeddings Arabic Matryoshka")
    print("=" * 60)
    
    # Vérifications préliminaires
    if not check_python_version():
        sys.exit(1)
    
    # Installation des dépendances
    if not install_dependencies():
        print("❌ Échec de l'installation des dépendances")
        sys.exit(1)
    
    # Test des imports
    if not test_imports():
        print("❌ Échec des tests d'import")
        sys.exit(1)
    
    # Vérification du dataset
    if not check_dataset():
        print("⚠️ Dataset non trouvé ou invalide")
        response = input("Voulez-vous créer un dataset d'exemple? (y/n): ")
        if response.lower() == 'y':
            if not create_sample_dataset():
                print("❌ Échec de la création du dataset")
                sys.exit(1)
        else:
            print("❌ Dataset requis pour continuer")
            sys.exit(1)
    
    # Test du système RAG
    if not test_rag_system():
        print("❌ Échec du test du système RAG")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Installation terminée avec succès!")
    print("\n📋 Prochaines étapes:")
    print("1. Lancez l'application: streamlit run app.py")
    print("2. Testez le système: python test_rag.py")
    print("3. Consultez la documentation: README_RAG.md")
    print("\n🎉 Le système RAG est prêt à être utilisé!")

if __name__ == "__main__":
    main() 