# Rapport d'Exploration, de Data Visualisation et de Pre-processing des Données
**Projet : Classification de Produits e-Commerce Rakuten France**  
**Cursus : Data Scientist** | **Étape : Rendu 1 (Deadline: Vendredi 11 septembre)**

---

## 1. Cadre du Projet & Présentation des Données

### 1.1 Objectifs Métier et Scientifiques
Le catalogage automatique des produits est un enjeu stratégique pour les marketplaces de commerce électronique comme Rakuten. Une classification précise des annonces permet d'optimiser l'expérience utilisateur via :
- La recherche personnalisée et le filtrage par facettes,
- La recommandation automatisée de produits complémentaires,
- La modération du catalogue et la détection d'erreurs de catégorisation par les vendeurs.

L'objectif de ce projet est de prédire le code type de catégorie (`prdtypecode`, parmi **27 classes distinctes**) à partir de données textuelles (titre commercial `designation` et texte descriptif `description`) et visuelles (image du produit `imageid` / `productid`).

### 1.2 Volumétrie et Structure du Jeu de Données
Le dataset officiel Rakuten France Multimodal comprend **98 728 produits** répartis en deux sous-ensembles :
- **Jeu d'entraînement (`X_train`)** : 84 916 lignes avec étiquettes de catégories (`Y_train`).
- **Jeu de test (`X_test`)** : 13 812 lignes sans étiquettes (destinées à l'évaluation finale).
- **Images associées** : 98 728 fichiers JPG au format fixe 500x500 pixels RGB (~2.55 Go décompressés).

| Variable | Type informatique | Disponibilité a priori | Taux de NA | Rôle & Remarques |
| :--- | :--- | :--- | :--- | :--- |
| `designation` | `object` (Texte / String) | Oui (Obligatoire) | **0.00 %** | Titre court du produit. Variable la plus informative pour la classification. Contient du bruit multilingue et du code d'encodage. |
| `description` | `object` (Texte / String) | Oui (Optionnel) | **35.09 %** (29 800 NA dans Train, 4 886 dans Test) | Description détaillée. Riche mais facultative. Contient des balises HTML (`<p>`, `<br>`, CSS style) et des entités non décodées. |
| `productid` | `int64` | Oui | **0.00 %** | Identifiant unique du produit. Utilisé pour faire la correspondance avec le fichier image. |
| `imageid` | `int64` | Oui | **0.00 %** | Identifiant unique du fichier photo. Associe l'image au produit (format `image_{imageid}_product_{productid}.jpg`). |
| `prdtypecode` | `int64` (Catégoriel) | Non (Variable Cible) | **0.00 %** (dans Train) | **Variable Cible à prédire** (27 classes uniques, de 10 à 2905). Implique l'utilisation du Weighted F1-Score. |

---

## 2. Pre-processing des Données

### 2.1 Nettoyage de la Donnée Textuelle
Le nettoyage du texte est une étape critique en NLP avant la vectorisation. Les opérations suivantes ont été appliquées via notre module dédié `src/preprocessing/text_cleaner.py` :
1. **Décodage HTML** : Conversion des entités HTML (ex: `&eacute;` $\rightarrow$ `é`, `&amp;` $\rightarrow$ `&`) avec `html.unescape`.
2. **Suppression des Balises HTML/CSS/JS** : Élimination complète des marqueurs de structure HTML (`<p>`, `<br>`, `<div>`, `<style>`, `<table>`) par expressions régulières (`re.sub`).
3. **Correction des Caractères Corrompus** : Suppression des caractères d'erreur Unicode (`\ufffd` ou `N`) provenant de mauvais décodages UTF-8 / CP1252 d'origine.
4. **Normalisation Minuscules & Ponctuation** : Conversion de l'ensemble des caractères en minuscules et remplacement des caractères spéciaux non alphanumériques par des espaces.
5. **Filtrage des Stop-Words** : Retrait des mots vides bilingues (français et anglais issus de NLTK) ainsi que des jetons de longueur inférieure ou égale à 1 caractère.
6. **Racinisation (Optionnelle)** : Intégration du `FrenchStemmer` de NLTK pour réduire les mots à leur racine contextuelle.

```python
# Exemple de transformation avant / après nettoyage :
# Avant : "<p>Journal Des Arts (Le) N 133 Du 28/09/2001 - L'art Et Son Ma&eacute;cenat</p>"
# Après : "journal arts 133 28 09 2001 art maécenat"
```

### 2.2 Traitement des Valeurs Manquantes (NaN)
La variable `description` présente un taux d'absence élevé (**35.09 %**).
- **Stratégie retenue** : Les valeurs manquantes `NaN` sont imputées par des chaînes vides `""`.
- **Création du champ fusionné `text_full`** : Concaténation de `designation_clean` et `description_clean`. Cette approche permet de tirer parti de la description complète lorsqu'elle est disponible, tout en garantissant un champ textuel unifié sans rupture d'information pour le TF-IDF et les modèles de langue.

### 2.3 Traitement des Données d'Images
Le module `src/preprocessing/image_processor.py` gère l'extraction et l'intégrité des images :
- **Décompression Automatique** : Extraction sécurisée du fichier `images.zip` dans le répertoire local `images/`.
- **Contrôle d'Intégrité** : 100 % des 98 728 images ont été vérifiées avec succès (aucun fichier corrompu ou illisible).
- **Transforms pour Deep Learning** : Définition des pipelines PyTorch / Torchvision (redimensionnement 224x224, normalisation ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`, data augmentation par rotation et retournement horizontal).

---

## 3. Ingénierie de Variables (Feature Engineering)

Afin de fournir des signaux discriminants aussi bien aux algorithmes de Gradient Boosting (LightGBM / XGBoost) qu'aux réseaux de neurones, nous avons créé un ensemble riche de nouvelles variables :

### 3.1 Features Statistiques et Métriques Textuelles
- **`desig_char_len` & `desc_char_len`** : Nombre total de caractères dans la désignation et la description.
- **`desig_word_count` & `desc_word_count`** : Nombre de mots dans le texte brut et le texte nettoyé.
- **`has_description`** : Variable binaire Indicateur (1 si description présente, 0 sinon). Très utile car le taux d'absence dépend fortement du type de produit (ex: les journaux ont rarement des descriptions, les meubles en ont presque toujours).
- **`desig_caps_count` & `desig_caps_ratio`** : Nombre et proportion de lettres majuscules dans la désignation (marqueur fort pour les acronymes, marques ou références de consoles/jeux).
- **`desig_digit_count` & `desig_digit_ratio`** : Nombre et proportion de chiffres (marqueur de numéros de tomes de manga, volumes de magazines, ou dimensions de meubles).
- **`desc_digit_ratio`** : Proportion de chiffres dans la description.

### 3.2 Feature Engineering NLP : Vectorisation TF-IDF
- Extraction d'une matrice sparse **TF-IDF de 5 000 caractéristiques** sur le champ `text_full`.
- Configuration : Unigrammes et Bigrammes (`ngram_range=(1, 2)`), ponderation sublinéaire de la fréquence des termes (`sublinear_tf=True`), fréquence minimale documentaire `min_df=3`.

### 3.3 Features Visuelles et Propriétés Statistiques des Images
Pour chaque produit, nous avons extrait les métadonnées statistiques directement à partir du fichier JPG :
- **`img_width`, `img_height`, `img_aspect_ratio`** : Dimensions physiques et ratio largeur/hauteur (stables à 500x500 et ratio 1.0).
- **`img_mean_r`, `img_mean_g`, `img_mean_b`** : Intensité moyenne de chaque canal de couleur RGB.
- **`img_std_r`, `img_std_g`, `img_std_b`** : Écart-type des canaux de couleur (marqueur de diversité chromatique).
- **`img_brightness`** : Luminosité moyenne de l'image (du blanc de fond de studio au sombre).
- **`img_contrast`** : Contraste global de l'image (écart-type des niveaux de gris).

---

## 4. Visualisations et Analyses Statistiques

### 4.1 Répartition des Classes Cibles (`prdtypecode`)
La variable cible présente un **déséquilibre de classe significatif** :
- **Classe maximale** : Code `2583` (Piscine & Spa) avec 10 209 exemples (**12.02 %** du jeu d'entraînement).
- **Classe minimale** : Code `1180` (Figurines Wargame) avec 764 exemples (**0.90 %** du jeu d'entraînement).
- **Ratio Déséquilibre Max/Min** : $13.36$.
- **Test du $\chi^2$** : $p < 0.0001$, confirmant un rejet massif de l'hypothèse d'uniformité.

> **Conséquence directe pour la modélisation** : Nécessité d'appliquer une stratégie de découpage stratifié (**Stratified K-Fold**) lors de la validation croisée et d'utiliser le **Weighted F1-score** comme métrique d'évaluation principale.

### 4.2 Distribution de la Longueur de Texte Avant vs Après Nettoyage
- **Désignation** : Longueur moyenne brute de ~75 caractères (~13.5 mots). Après suppression des stop-words et de la ponctuation, la longueur moyenne nettoyée passe à ~8.2 mots informatifs.
- **Description** : Quand elle est présente, la longueur moyenne brute est de ~120 mots (médiane ~60 mots). Le nettoyage HTML réduit le nombre de caractères de ~22% tout en conservant 100% de la sémantique utile.

---

## 5. Structuration du Répertoire GitHub

Conformément au template fourni par DataScientest, le répertoire du projet est organisé de manière modulaire et professionnelle :

```
LIORA RAKUTEN/
│
├── data/                       # Emplacement des fichiers CSV et images décompressées
├── src/                        # Code source Python modulaire
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py     # Nettoyage HTML, Unicode, stop-words, stemmer
│   │   ├── feature_engineering.py # Extraction des métriques textuelles & TF-IDF
│   │   └── image_processor.py  # Validation JPG, extraction stats RGB, PyTorch transforms
│   └── utils/
│       └── data_loader.py      # Chargeur unifié et mapping des 27 catégories
│
├── scripts/
│   └── run_preprocessing.py    # Script CLI d'exécution bout-en-bout
│
├── notebooks/
│   └── 01_exploration_and_preprocessing.ipynb # Notebook interactif documenté
│
├── outputs/                    # Features extraites (CSV, Parquet, matrices TF-IDF)
│   ├── train_features.csv
│   ├── test_features.csv
│   ├── tfidf_train.npz
│   ├── tfidf_test.npz
│   └── image_meta_train.csv
│
├── reports/                    # Rapports de synthèse
│   └── rapport_preprocessing.md
│
├── Classification de produits e-commerce Rakuten.pdf
├── Projets_méthodologie_rapports.pdf
└── Template - Rapport exploration des données.xlsx
```

---

## 6. Conclusion et Directives pour l'Étape 3 (Modélisation)

Le processus de pré-traitement et de feature engineering mené lors de cette **Étape 2** fournit une base de données propre, enrichie et prête pour les expérimentations avancées de Machine Learning et Deep Learning :

1. **Jeu de données Texte révisé** : Suppression de 100% des bruits HTML et d'encodage, unification des champs dans `text_full` sans aucune perte d'échantillons.
2. **Matrice de Features Multimodales** : Combinaison des métriques textuelles, des propriétés visuelles RGB et de la représentation TF-IDF (5 000 dimensions).
3. **Pistes pour l'Étape 3 (Modélisation)** :
   - **Baseline Tabulaire / NLP** : Modèles LightGBM et CatBoost entraînés sur la combinaison `features statistiques + TF-IDF` (F1-score cible $\ge 0.81$).
   - **Baseline Vision** : Fine-tuning d'un backbone ResNet50 ou EfficientNet-B0 pré-entraîné sur ImageNet pour prédire la catégorie à partir des images seules.
   - **Fusion Multimodale (Late / Early Fusion)** : Concaténation des représentations textuelles (embeddings CamemBERT) et visuelles (features CNN) via une tête de classification dense pour viser le haut du classement benchmark.
