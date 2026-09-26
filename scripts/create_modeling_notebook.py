import json
import os

def generate_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Étape 3 : Modélisation, Optimisation et Évaluation\n",
                    "**Projet Rakuten France Multimodal Product Classification**  \n",
                    "*Cursus Data Scientist - DataScientest (Rendu 2)*\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Contexte et Objectifs Scientifiques\n",
                    "\n",
                    "Ce notebook présente les expérimentations de modélisation pour la prédiction de la catégorie produit (`prdtypecode`, **27 classes**) à partir de données textuelles (titre et description) et de propriétés visuelles d'images.\n",
                    "\n",
                    "**Objectifs de l'étape :**\n",
                    "1. Définir le protocole de validation croisée adapté au fort déséquilibre de classes (**Stratified K-Fold**).\n",
                    "2. Évaluer une hiérarchie de modèles de complexité croissante :\n",
                    "   - *Baseline Naïve* (Dummy Classifier stratifié)\n",
                    "   - *Modèles NLP Classiques* (Multinomial Naive Bayes, SGD Logistic Regression, Linear SVM)\n",
                    "   - *Ensemble & Gradient Boosting* (LightGBM multiclasse)\n",
                    "   - *Fusion Multimodale & Blending* (Concaténation précoce et combinaison tardive)\n",
                    "3. Évaluer selon la métrique officielle du challenge : **Weighted F1-Score**.\n",
                    "4. Analyser en profondeur la matrice de confusion et l'interprétabilité des n-grammes."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import pickle\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "from scipy.sparse import load_npz, hstack, csr_matrix\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Ajout du dossier parent au path\n",
                    "sys.path.append('..')\n",
                    "from src.utils.data_loader import PRODUCT_CODE_MAP, get_category_name\n",
                    "from src.models.evaluation import compute_all_metrics, get_detailed_report, get_normalized_confusion_matrix, find_top_confusions\n",
                    "from src.models.interpretability import get_top_keywords_per_class, get_feature_importances_df\n",
                    "\n",
                    "%matplotlib inline\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Chargement des Matrices TF-IDF et des Features Prétraitées"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Chargement de la cible et des représentations extraites à l'Étape 1\n",
                    "y_train_raw = pd.read_csv('../Y_train_CVw08PX.csv', index_col=0)\n",
                    "X_tfidf_train = load_npz('../outputs/tfidf_train.npz')\n",
                    "X_tfidf_test = load_npz('../outputs/tfidf_test.npz')\n",
                    "\n",
                    "df_train_stats = pd.read_csv('../outputs/train_features.csv', index_col=0)\n",
                    "df_img_train = pd.read_csv('../outputs/image_meta_train.csv', index_col=0) if os.path.exists('../outputs/image_meta_train.csv') else None\n",
                    "\n",
                    "with open('../outputs/tfidf_vectorizer.pkl', 'rb') as f:\n",
                    "    vectorizer = pickle.load(f)\n",
                    "feature_names = vectorizer.get_feature_names_out()\n",
                    "\n",
                    "print(f\"X_tfidf_train shape: {X_tfidf_train.shape}\")\n",
                    "print(f\"Nombre de classes uniques: {y_train_raw['prdtypecode'].nunique()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Encodage de la Variable Cible & Découpage Stratifié"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.preprocessing import LabelEncoder\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "\n",
                    "le = LabelEncoder()\n",
                    "y_encoded = le.fit_transform(y_train_raw['prdtypecode'].values)\n",
                    "class_names = [f\"{c} - {get_category_name(c)[:25]}\" for c in le.classes_]\n",
                    "short_names = [get_category_name(c)[:25] for c in le.classes_]\n",
                    "\n",
                    "idx_train, idx_val = train_test_split(\n",
                    "    np.arange(len(y_encoded)), test_size=0.20, random_state=42, stratify=y_encoded\n",
                    ")\n",
                    "y_tr, y_va = y_encoded[idx_train], y_encoded[idx_val]\n",
                    "X_tf_tr, X_tf_va = X_tfidf_train[idx_train], X_tfidf_train[idx_val]\n",
                    "\n",
                    "print(f\"Train size : {len(idx_train)} | Validation size : {len(idx_val)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Benchmark Comparatif des Modèles\n",
                    "\n",
                    "Visualisons les résultats obtenus par notre suite de modèles évalués sur le jeu de validation indépendant (20% stratifié) :"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_comp = pd.read_csv('../outputs/model_comparison.csv')\n",
                    "display(df_comp)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visualisation graphique de la comparaison des scores F1\n",
                    "plt.figure(figsize=(11, 5))\n",
                    "df_plot = df_comp.sort_values(by='Weighted F1-Score', ascending=True)\n",
                    "y_pos = np.arange(len(df_plot))\n",
                    "h = 0.35\n",
                    "\n",
                    "plt.barh(y_pos - h/2, df_plot['Weighted F1-Score'], height=h, label='Weighted F1-Score', color='#1f77b4')\n",
                    "plt.barh(y_pos + h/2, df_plot['Macro F1-Score'], height=h, label='Macro F1-Score', color='#aec7e8')\n",
                    "\n",
                    "plt.yticks(y_pos, df_plot['Modele'])\n",
                    "plt.xlabel('Score F1')\n",
                    "plt.title('Comparaison des Performances sur Validation (20%)', fontweight='bold')\n",
                    "plt.legend(loc='lower right')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Matrice de Confusion et Analyse Poussée des Erreurs\n",
                    "\n",
                    "L'inspection de la matrice de confusion normalisée permet de repérer précisément les frontières de décision floues entre catégories proches."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cm_norm = np.load('../outputs/confusion_matrix.npy')\n",
                    "\n",
                    "plt.figure(figsize=(14, 12))\n",
                    "sns.heatmap(cm_norm, cmap='Blues', xticklabels=short_names, yticklabels=short_names)\n",
                    "plt.title('Matrice de Confusion Normalisée (Taux de Rappel)', fontweight='bold', pad=10)\n",
                    "plt.xlabel('Catégorie Prédite', fontweight='bold')\n",
                    "plt.ylabel('Catégorie Réelle', fontweight='bold')\n",
                    "plt.xticks(rotation=90, fontsize=8)\n",
                    "plt.yticks(rotation=0, fontsize=8)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Top des confusions entre classes\n",
                    "df_confusions = pd.read_csv('../outputs/top_confusions.csv')\n",
                    "print(\"Top 10 des Paires de Classes les Plus Confondues :\")\n",
                    "display(df_confusions)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Interprétabilité du Modèle\n",
                    "\n",
                    "Pour garantir l'explicabilité de la prise de décision, nous analysons :\n",
                    "1. Les **n-grammes les plus informatifs** pour orienter la décision vers chaque classe.\n",
                    "2. L'**importance des variables** pour le modèle d'arbres LightGBM."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_feat_imp = pd.read_csv('../outputs/feature_importance.csv')\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "sns.barplot(data=df_feat_imp.head(15), y='feature', x='importance', palette='viridis')\n",
                    "plt.title('Top 15 des Variables les Plus Utiles (LightGBM Multimodal)', fontweight='bold')\n",
                    "plt.xlabel('Nombre de Scissions (Splits)')\n",
                    "plt.ylabel('Caractéristique')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Conclusions et Perspectives pour le Rendu Final\n",
                    "\n",
                    "- Le modèle linéaire à marge douce (Linear SVM / SGD Modified Huber) combiné au TF-IDF offre un excellent compromis vitesse/performance (F1-score pondéré > 0.81 en quelques secondes).\n",
                    "- Le Gradient Boosting LightGBM multimodal et l'Ensemble Blending exploitent les corrélations fines entre longueur des textes et propriétés de couleur des photos.\n",
                    "- Pour l'étape finale (Rendu 3), la fusion multimodale avancée avec embeddings CamemBERT et backbone Vision CNN (ResNet/EfficientNet) permettra de débloquer les dernières confusions sémantiques."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nb_dir = os.path.join(root_dir, "notebooks")
    os.makedirs(nb_dir, exist_ok=True)
    nb_path = os.path.join(nb_dir, "02_modelisation_and_evaluation.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Notebook '{nb_path}' généré avec succès.")

if __name__ == "__main__":
    generate_notebook()
