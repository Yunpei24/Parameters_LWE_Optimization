# 📊 DataCollector & Visualisation - Guide Complet

## ✅ Fonctionnalités Implémentées

### 1. DataCollector (✅ Implémenté)

Le système de collecte de données est complètement intégré dans `model.py` :

#### Métriques au niveau du modèle
```python
model_reporters={
    "Global_Best_Fitness": lambda m: m.global_best_fitness,
    "Global_Best_N": lambda m: m.global_best_params['n'],
    "Global_Best_Q": lambda m: m.global_best_params['q'],
    "Global_Best_Sigma": lambda m: m.global_best_params['sigma'],
    "Average_Fitness": self.compute_average_fitness,
    "Diversity": self.compute_diversity,
    "Convergence_Rate": self.compute_convergence_rate
}
```

#### Métriques au niveau des agents
```python
agent_reporters={
    "Fitness": "fitness_personal",
    "N": lambda a: a.current_params['n'],
    "Q": lambda a: a.current_params['q'],
    "Sigma": lambda a: a.current_params['sigma']
}
```

### 2. Plots Matplotlib (✅ Implémenté)

Le fichier `run.py` génère automatiquement 6 graphiques :

1. **Fitness Evolution** - Évolution de la meilleure fitness et moyenne
2. **Population Diversity** - Diversité des agents dans l'espace des paramètres
3. **Convergence Rate** - Pourcentage d'agents convergés
4. **Best Lattice Dimension (n)** - Évolution du meilleur paramètre n
5. **Best Modulus (q)** - Évolution du meilleur paramètre q
6. **Best Noise (σ)** - Évolution du meilleur paramètre sigma

**Utilisation :**
```bash
python run.py
# Génère automatiquement 'optimization_results.png'
```

### 3. Visualisation Interactive Solara (✅ Implémenté - NOUVEAU!)

Interface web moderne avec contrôles en temps réel dans `app.py` :

#### Fonctionnalités principales :

##### Contrôles Interactifs
- 🔢 **Sliders** pour ajuster les paramètres :
  - Nombre d'explorateurs (5-50)
  - Poids de sécurité α (0.0-1.0)
  - Poids de performance β (0.0-1.0)
  - Nombre max de steps (10-500)
  
- 🌐 **Sélecteur de topologie** :
  - Ring (anneau)
  - Random (aléatoire)
  - All (tous connectés)

- 🎮 **Boutons de contrôle** :
  - **Reset** : Réinitialiser avec nouveaux paramètres
  - **Step** : Exécuter une seule itération
  - **Run/Pause** : Lancer/arrêter l'exécution automatique

##### Visualisations en Temps Réel
1. **Carte des meilleurs paramètres** avec :
   - Dimension (n)
   - Modulus (q)
   - Noise (σ)
   - Niveau de sécurité (bits)
   - Coût de performance
   - Fitness et convergence

2. **Graphiques dynamiques** :
   - Évolution de la fitness (global + moyenne)
   - Diversité de la population
   - Taux de convergence
   - Évolution des 3 paramètres (n, q, σ)

##### Lancement
```bash
solara run app.py
# Ouvrir http://localhost:8765/ dans le navigateur
```

## 🎯 Comparaison des 3 Modes de Visualisation

| Fonctionnalité | run.py | server.py | app.py (Solara) |
|----------------|---------|-----------|-----------------|
| Type | CLI + Plots statiques | Mesa Web | Solara Web moderne |
| Interface | Terminal | Navigateur (classique) | Navigateur (moderne) |
| Temps réel | ❌ | ✅ | ✅ |
| Contrôles interactifs | ❌ | ✅ | ✅✅ (plus riches) |
| Graphiques | Matplotlib (PNG) | Charts Mesa | Matplotlib interactif |
| Step-by-step | ❌ | ✅ | ✅ |
| Responsive design | N/A | ⚠️ Basique | ✅ Moderne |
| Export données | ✅ (PNG) | ❌ | ✅ (via matplotlib) |

## 📈 Métriques Disponibles

### Métriques de Performance
- **Global Best Fitness** : Meilleure fitness trouvée globalement
- **Average Fitness** : Fitness moyenne de tous les agents
- **Convergence Rate** : % d'agents dans 10% de l'optimum

### Métriques de Diversité
- **Diversity** : Écart-type des positions des agents (paramètre n)
- Indique si les agents explorent ou convergent

### Paramètres Optimaux
- **n** : Dimension du lattice
- **q** : Modulus
- **σ** : Écart-type du bruit gaussien

### Métriques Cryptographiques
- **Security Level** : Sécurité estimée en bits
- **Performance Cost** : Coût computationnel normalisé

## 🚀 Guide d'Utilisation Rapide

### Pour une analyse rapide
```bash
python run.py
# Génère les graphiques PNG automatiquement
```

### Pour expérimentation interactive (classique)
```bash
python server.py
# Naviguer vers http://127.0.0.1:8521/
```

### Pour expérimentation interactive (moderne) ⭐
```bash
solara run app.py
# Naviguer vers http://localhost:8765/
```

### Pour comparaison de topologies
```python
from run import compare_topologies
results = compare_topologies(n_steps=100, n_runs=5)
```

### Pour analyse de sensibilité
```python
from run import sensitivity_analysis
results = sensitivity_analysis()
```

## 📊 Exemple de Workflow

1. **Exploration initiale** avec Solara (`app.py`)
   - Tester différentes configurations
   - Observer la convergence en temps réel
   - Ajuster α et β pour le trade-off sécurité/performance

2. **Analyse comparative** avec `run.py`
   - Lancer `compare_topologies()` pour comparer ring/random/all
   - Générer des graphiques comparatifs

3. **Étude de sensibilité**
   - Utiliser `sensitivity_analysis()` pour explorer l'impact de α
   - Identifier la configuration optimale

4. **Production finale**
   - Configurer le modèle avec les meilleurs paramètres
   - Exécuter plusieurs runs avec différentes seeds
   - Collecter les résultats pour analyse statistique

## 📝 Notes Importantes

### Avantages du DataCollector Mesa
- ✅ Collecte automatique à chaque step
- ✅ Stockage dans DataFrame pandas
- ✅ Facile à exporter (CSV, Excel)
- ✅ Compatible avec matplotlib, seaborn, plotly

### Pourquoi Solara ?
- ✅ Plus moderne que Mesa visualization
- ✅ Meilleures performances
- ✅ UI/UX supérieure
- ✅ Support de composants réactifs
- ✅ Facilité de déploiement

## 🎓 Ressources

- Documentation Mesa DataCollector: https://mesa.readthedocs.io/
- Documentation Solara: https://solara.dev/
- Matplotlib Gallery: https://matplotlib.org/stable/gallery/

---

**Résumé : Toutes les fonctionnalités de DataCollector et de visualisation sont complètement implémentées et fonctionnelles !** ✅
