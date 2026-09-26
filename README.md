# Projet Rakuten France - Classification Multimodale de Produits

Projet de classification supervisée multiclasse (27 catégories) combinant texte (`designation`, `description`) et images (photos 500x500 pixels).

---

## 📁 Structure du Répertoire

```
LIORA RAKUTEN/
│
├── src/                               # Code source modulaire Python
│   ├── preprocessing/                 # Pipelines de nettoyage texte, stats & images
│   │   ├── text_cleaner.py            # Nettoyage HTML, regex, stopwords
│   │   ├── feature_engineering.py     # Extraction de métriques et TF-IDF
│   │   └── image_processor.py         # Propriétés chromatiques RGB et intégrité
│   ├── models/                        # Modèles de modélisation (ML & Multimodal)
│   │   ├── baseline_models.py         # Dummy, Naive Bayes, SGD LogLoss, SVM Huber
│   │   ├── gradient_boosting.py       # Configuration LightGBM multiclasse
│   │   ├── multimodal.py              # Early & Late Fusion (sparse + dense)
│   │   ├── evaluation.py              # Weighted F1, classification report, confusions
│   │   └── interpretability.py        # Top n-grammes discriminants et importances
│   └── utils/
│       └── data_loader.py             # Chargeur de données et PRODUCT_CODE_MAP (27 classes)
│
├── scripts/                           # Scripts d'exécution autonomes
│   ├── run_preprocessing.py           # Pipeline complet Rendu 1 (Nettoyage & TF-IDF)
│   ├── run_modeling.py                # Pipeline complet Rendu 2 (Benchmark & Graphiques)
│   ├── generate_plots.py              # Visualisations d'exploration
│   ├── create_notebook.py             # Générateur notebook 01
│   └── create_modeling_notebook.py    # Générateur notebook 02
│
├── notebooks/                         # Notebooks Jupyter interactifs documentés
│   ├── 01_exploration_and_preprocessing.ipynb # Rendu 1 : Exploration & Preprocessing
│   └── 02_modelisation_and_evaluation.ipynb   # Rendu 2 : Modélisation & Évaluation
│
├── outputs/                           # Artefacts et sorties calculées
│   ├── best_model.pkl                 # Modèle SGD Linear SVM sauvegardé
│   ├── lgb_multimodal.pkl             # Modèle LightGBM Multimodal sauvegardé
│   ├── label_encoder.pkl              # Encodage des 27 classes (0 à 26)
│   ├── tfidf_vectorizer.pkl           # Vectoriseur TF-IDF (5 000 dimensions)
│   ├── model_comparison.csv           # Tableau comparatif des performances
│   ├── classification_report.csv      # Métriques détaillées par classe
│   ├── top_confusions.csv             # Paires de classes les plus confondues
│   ├── y_test_predictions.csv         # Prédictions test prêtes pour soumission
│   └── fig1_*.png à fig10_*.png       # Visualisations haute résolution
│
├── reports/                           # Rapports de synthèse officiels DataScientest
│   ├── rapport_preprocessing.md       # Rendu 1 : Exploration & Pre-processing
│   └── rapport_modelisation.md        # Rendu 2 : Rapport de Modélisation
│
├── requirements.txt                   # Dépendances Python du projet
└── README.md                          # Documentation d'accueil du projet
```

---

## 🚀 Installation Rapide

```bash
# 1. Cloner le dépôt et créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Linux/Mac
.venv\Scripts\activate     # Sur Windows

# 2. Installer les dépendances
pip install -r requirements.txt
```

---

## 📊 Résumé des Performances (Phase 2 - Modélisation)

- **Meilleur Modèle Multimodal** : **LightGBM Multimodal** $\rightarrow$ **F1-Score pondéré = 0.8233** (Log Loss = 0.5922)
- **Meilleur Modèle Temps Réel** : **SGD Linear SVM Huber** $\rightarrow$ **F1-Score pondéré = 0.7797** (Inférence < 10 ms)
- **Validation Croisée Stratifiée (5-plis)** : Score moyen = **0.7791 $\pm$ 0.0024**
