import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

def get_dummy_model(strategy: str = "stratified", random_state: int = 42):
    """
    Retourne un DummyClassifier (baseline plancher).
    Strategies: 'stratified' (respecte les fréquences a priori) ou 'most_frequent'.
    """
    return DummyClassifier(strategy=strategy, random_state=random_state)

def get_naive_bayes_model(alpha: float = 0.1):
    """
    Retourne un Multinomial Naive Bayes adapté aux fréquences de termes TF-IDF.
    """
    return MultinomialNB(alpha=alpha)

def get_logistic_regression_model(C: float = 1.0, max_iter: int = 200, random_state: int = 42):
    """
    Retourne une Régression Logistique Multiclasse (solver lbfgs).
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )

def get_sgd_logloss_model(alpha: float = 1e-5, max_iter: int = 100, random_state: int = 42):
    """
    Retourne un classifieur linéaire entraîné par Descente de Gradient Stochastique (SGD)
    avec fonction de perte log-loss (régression logistique à grande échelle).
    Ultra-rapide sur matrices creuses TF-IDF avec estimation de probabilités.
    """
    return SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        penalty="l2",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )

def get_sgd_svm_model(alpha: float = 1e-5, max_iter: int = 100, random_state: int = 42):
    """
    Retourne un Support Vector Machine Linéaire (SVM) entraîné par SGD (loss='modified_huber').
    La perte modified_huber offre les marges d'un SVM tout en fournissant des probabilités calibrées.
    """
    return SGDClassifier(
        loss="modified_huber",
        alpha=alpha,
        penalty="l2",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )

def get_random_forest_model(n_estimators: int = 100, max_depth: int = 25, random_state: int = 42):
    """
    Retourne une Forêt Aléatoire (Bagging).
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
