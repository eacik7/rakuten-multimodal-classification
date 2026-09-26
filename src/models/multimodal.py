import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse, csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

class MultimodalEarlyFusion:
    """
    Gestionnaire d'Early Fusion (concaténation au niveau des features).
    Combine les matrices TF-IDF creuses avec les features tabulaires et visuelles standardisées.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_tfidf, X_tabular):
        """Standardise les features tabulaires/visuelles et les concatène au TF-IDF."""
        X_tab_scaled = self.scaler.fit_transform(X_tabular)
        X_tab_sparse = csr_matrix(X_tab_scaled)
        return hstack([X_tfidf, X_tab_sparse], format="csr")

    def transform(self, X_tfidf, X_tabular):
        """Applique la transformation apprise sur les nouvelles données."""
        X_tab_scaled = self.scaler.transform(X_tabular)
        X_tab_sparse = csr_matrix(X_tab_scaled)
        return hstack([X_tfidf, X_tab_sparse], format="csr")

class MultimodalLateFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifieur de fusion tardive (Late Fusion / Blending probabiliste).
    Combine les probabilités d'un modèle textuel NLP et d'un modèle tabulaire/visuel.
    """
    def __init__(self, text_model, vision_tabular_model, weight_text: float = 0.85):
        self.text_model = text_model
        self.vision_tabular_model = vision_tabular_model
        self.weight_text = weight_text
        self.scaler = StandardScaler()
        self.classes_ = None

    def fit(self, X_tfidf, X_tab, y):
        self.classes_ = np.unique(y)
        # Entraînement du modèle texte sur TF-IDF
        self.text_model.fit(X_tfidf, y)
        # Entraînement du modèle tabulaire & vision
        X_tab_scaled = self.scaler.fit_transform(X_tab)
        self.vision_tabular_model.fit(X_tab_scaled, y)
        return self

    def predict_proba(self, X_tfidf, X_tab):
        proba_text = self.text_model.predict_proba(X_tfidf)
        X_tab_scaled = self.scaler.transform(X_tab)
        proba_tab = self.vision_tabular_model.predict_proba(X_tab_scaled)
        
        # Combinaison pondérée convexe
        proba_final = self.weight_text * proba_text + (1.0 - self.weight_text) * proba_tab
        return proba_final

    def predict(self, X_tfidf, X_tab):
        proba = self.predict_proba(X_tfidf, X_tab)
        return self.classes_[np.argmax(proba, axis=1)]
