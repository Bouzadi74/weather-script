# Guide de Dépannage - Timeouts Ollama

## 🚨 Problème : Timeout avec Ollama

### Symptômes
```
Error generating script with Ollama: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=180)
```

### ✅ Solutions Implémentées

#### 1. Prompt Raccourci
Le fichier `prompt_weather.txt` a été optimisé :
- **Avant** : ~2000 caractères
- **Après** : ~500 caractères
- **Réduction** : 75% de la taille

#### 2. Contexte RAG Optimisé
- **Nombre d'exemples** : Réduit de 3 à 2
- **Longueur des exemples** : Limité à 100 caractères
- **Format** : Simplifié et concis

#### 3. Paramètres de Recherche
```python
# Dans app.py
top_k=2  # Au lieu de 3

# Dans rag_integration.py
context_summary = "\n".join([f"- {context[:100]}..." for context in rag_context[:2]])
```

### 🔧 Optimisations Supplémentaires

#### 1. Vérifier la Configuration Ollama
```bash
# Vérifier que Ollama fonctionne
curl http://localhost:11434/api/tags

# Tester avec un prompt simple
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "Bonjour, comment allez-vous?",
    "stream": false
  }'
```

#### 2. Augmenter le Timeout (si nécessaire)
Dans `script.py`, vous pouvez augmenter le timeout :
```python
# Augmenter le timeout si nécessaire
response = requests.post(url, json=data, timeout=300)  # 5 minutes
```

#### 3. Réduire la Complexité du Modèle
Si les timeouts persistent, utilisez un modèle plus léger :
```python
# Dans script.py, changer le modèle
data = {
    "model": "llama2:7b",  # Modèle plus léger
    "prompt": prompt,
    "stream": False
}
```

### 📊 Monitoring des Performances

#### Test de Performance
```bash
python test_optimized_rag.py
```

#### Métriques à Surveiller
- **Temps de génération** : < 60 secondes
- **Longueur du prompt** : < 1500 caractères
- **Nombre d'exemples RAG** : 2 maximum

### 🚨 Dépannage Avancé

#### 1. Problème de Mémoire
```bash
# Vérifier l'utilisation mémoire
htop
# ou
free -h

# Redémarrer Ollama si nécessaire
sudo systemctl restart ollama
```

#### 2. Problème de Réseau
```bash
# Tester la connectivité
ping localhost
telnet localhost 11434
```

#### 3. Problème de Modèle
```bash
# Lister les modèles disponibles
ollama list

# Télécharger un modèle plus léger
ollama pull llama2:7b
```

### 🔄 Fallback Automatique

Le système inclut un fallback automatique :
```python
try:
    # Tentative avec RAG
    enhanced_prompt = rag_generator.enhance_prompt_with_rag(...)
except Exception as e:
    # Fallback vers prompt simple
    enhanced_prompt = prompt.replace("{data}", weather_data)
```

### 📈 Optimisations Futures

#### 1. Cache des Embeddings
```python
# Sauvegarder les embeddings
rag_system.save_embeddings("embeddings_cache.npy")

# Charger les embeddings
rag_system.load_embeddings("embeddings_cache.npy")
```

#### 2. Quantification du Modèle
```bash
# Utiliser un modèle quantifié
ollama pull llama2:7b-q4_0
```

#### 3. Parallélisation
```python
# Traitement parallèle des embeddings
from concurrent.futures import ThreadPoolExecutor
```

### ✅ Checklist de Vérification

- [ ] Ollama fonctionne sur le port 11434
- [ ] Le modèle est téléchargé et accessible
- [ ] La mémoire système est suffisante
- [ ] Le prompt est < 1500 caractères
- [ ] Le nombre d'exemples RAG est ≤ 2
- [ ] Le timeout est configuré à 180s minimum

### 🆘 En Cas d'Urgence

Si les timeouts persistent :

1. **Désactiver temporairement le RAG** :
   ```python
   # Dans app.py, commenter la section RAG
   # enhanced_prompt = st.session_state.rag_generator.enhance_prompt_with_rag(...)
   enhanced_prompt = prompt.replace("{data}", st.session_state.extracted_text)
   ```

2. **Utiliser un prompt minimal** :
   ```python
   minimal_prompt = f"أنت مذيع نشرة جوية. {weather_data}"
   ```

3. **Augmenter les ressources** :
   - Plus de RAM
   - CPU plus puissant
   - Modèle plus léger

### 📞 Support

En cas de problème persistant :
1. Vérifiez les logs Ollama : `ollama logs`
2. Testez avec un prompt simple
3. Vérifiez la configuration système
4. Consultez la documentation Ollama

---

**Note** : Ces optimisations maintiennent la qualité tout en réduisant significativement les risques de timeout. 