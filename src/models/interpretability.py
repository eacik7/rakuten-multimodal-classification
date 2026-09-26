import numpy as np
import pandas as pd

def get_top_keywords_per_class(linear_model, feature_names, class_names, top_n: int = 8):
    """
    Extrait pour chaque catégorie les n-grammes TF-IDF ayant les coefficients positifs
    les plus élevés (impact direct sur la log-cote d'appartenance à la classe).
    """
    coefs = linear_model.coef_
    results = {}
    for idx, class_label in enumerate(class_names):
        top_indices = np.argsort(coefs[idx])[-top_n:][::-1]
        top_words = [(feature_names[i], round(float(coefs[idx][i]), 3)) for i in top_indices]
        results[class_label] = top_words
    return results

def get_feature_importances_df(tree_model, feature_names, top_n: int = 25):
    """
    Extrait l'importance des variables (Gain / Impureté Gini) pour les modèles d'arbres
    (Random Forest, LightGBM).
    """
    importances = tree_model.feature_importances_
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df_imp.head(top_n)

def explain_single_prediction(linear_model, tfidf_row, feature_names, class_idx, top_n: int = 5):
    """
    Décompose la prédiction pour un produit donné en affichant les mots qui ont le plus
    contribué au score de la classe prédite.
    """
    row_dense = tfidf_row.toarray().flatten() if hasattr(tfidf_row, "toarray") else np.array(tfidf_row).flatten()
    nonzero_idx = np.where(row_dense > 0)[0]
    contributions = []
    for idx in nonzero_idx:
        w_val = row_dense[idx]
        c_val = linear_model.coef_[class_idx, idx]
        contributions.append((feature_names[idx], w_val * c_val))
    
    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:top_n]
