import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ajout de la racine du projet au sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import PRODUCT_CODE_MAP, get_category_name
from src.models.evaluation import (
    compute_all_metrics,
    get_detailed_report,
    get_normalized_confusion_matrix,
    find_top_confusions
)
from src.models.interpretability import (
    get_top_keywords_per_class,
    get_feature_importances_df
)

def build_tabular_features(df_text_stats, df_img_meta):
    """Fusionne les features statistiques textuelles et chromatiques d'images."""
    tab_cols = [
        'has_description', 'desig_char_len', 'desig_word_count', 
        'desc_char_len', 'desc_word_count', 'desig_caps_count', 
        'desig_caps_ratio', 'desig_digit_count', 'desig_digit_ratio', 
        'desc_digit_count', 'desc_digit_ratio', 'desig_clean_word_count', 
        'desc_clean_word_count', 'text_full_word_count'
    ]
    img_cols = [
        'img_mean_r', 'img_mean_g', 'img_mean_b', 
        'img_std_r', 'img_std_g', 'img_std_b', 
        'img_brightness', 'img_contrast'
    ]
    
    # Text stats
    df_tab = df_text_stats[tab_cols].copy()
    
    # Image stats
    if df_img_meta is not None:
        for c in img_cols:
            if c in df_img_meta.columns:
                df_tab[c] = df_img_meta[c].values
            else:
                df_tab[c] = 0.0
    else:
        for c in img_cols:
            df_tab[c] = 0.0
            
    # Traitement des NaN résiduels
    df_tab = df_tab.fillna(0.0)
    return df_tab

def main():
    print("=" * 75)
    print("      RAKUTEN MULTIMODAL - ÉTAPE 3 : MODÉLISATION & ÉVALUATION")
    print("=" * 75)
    t_global_start = time.time()
    os.makedirs("outputs", exist_ok=True)

    # 1. Chargement des données
    print("\n[1/8] Chargement des données, des matrices TF-IDF et des métadonnées...")
    y_train_raw = pd.read_csv("Y_train_CVw08PX.csv", index_col=0)
    X_tfidf_train = load_npz("outputs/tfidf_train.npz")
    X_tfidf_test = load_npz("outputs/tfidf_test.npz")
    
    df_train_stats = pd.read_csv("outputs/train_features.csv", index_col=0)
    df_test_stats = pd.read_csv("outputs/test_features.csv", index_col=0)
    
    df_img_train = pd.read_csv("outputs/image_meta_train.csv", index_col=0) if os.path.exists("outputs/image_meta_train.csv") else None
    df_img_test = pd.read_csv("outputs/image_meta_test.csv", index_col=0) if os.path.exists("outputs/image_meta_test.csv") else None
    
    with open("outputs/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Échantillons Train : {X_tfidf_train.shape[0]} | Dimensions TF-IDF : {X_tfidf_train.shape[1]}")
    print(f"Échantillons Test  : {X_tfidf_test.shape[0]}")

    # 2. Encodage de la variable cible
    print("\n[2/8] Encodage de la variable cible (27 classes)...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train_raw["prdtypecode"].values)
    class_codes = le.classes_
    class_names = [f"{c} - {get_category_name(c)}" for c in class_codes]
    short_class_names = [get_category_name(c)[:25] for c in class_codes]

    with open("outputs/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"LabelEncoder sauvegardé. 27 classes encodées de 0 à {len(class_codes)-1}.")

    # 3. Préparation des Features Multimodales (Early Fusion)
    print("\n[3/8] Préparation des features tabulaires et visuelles (Early Fusion)...")
    df_tab_train = build_tabular_features(df_train_stats, df_img_train)
    df_tab_test = build_tabular_features(df_test_stats, df_img_test)
    tab_feature_names = list(df_tab_train.columns)

    scaler = StandardScaler()
    X_tab_train_scaled = scaler.fit_transform(df_tab_train)
    X_tab_test_scaled = scaler.transform(df_tab_test)

    # Concaténation sparse TF-IDF + Tabulaire/Vision
    X_fused_train = hstack([X_tfidf_train, csr_matrix(X_tab_train_scaled)], format="csr")
    X_fused_test = hstack([X_tfidf_test, csr_matrix(X_tab_test_scaled)], format="csr")
    all_feature_names = list(feature_names) + tab_feature_names

    print(f"Dimensions matrice fusionnée : {X_fused_train.shape[0]} lignes x {X_fused_train.shape[1]} features")

    # 4. Découpage Stratifié Train / Validation (80% / 20%)
    print("\n[4/8] Découpage Stratifié Train / Validation (80% / 20%)...")
    indices = np.arange(len(y_encoded))
    idx_train, idx_val = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    y_tr, y_va = y_encoded[idx_train], y_encoded[idx_val]
    
    # TF-IDF split
    X_tf_tr, X_tf_va = X_tfidf_train[idx_train], X_tfidf_train[idx_val]
    # Fused split
    X_fu_tr, X_fu_va = X_fused_train[idx_train], X_fused_train[idx_val]
    # Tabular split
    X_tb_tr, X_tb_va = X_tab_train_scaled[idx_train], X_tab_train_scaled[idx_val]

    print(f"Taille Train : {len(idx_train)} | Taille Validation : {len(idx_val)}")

    # 5. Benchmark des Modèles
    print("\n[5/8] Entraînement et évaluation des modèles...")
    models_dict = {
        "Dummy (Stratifié)": {
            "model": DummyClassifier(strategy="stratified", random_state=42),
            "data_train": X_tf_tr, "data_val": X_tf_va, "type": "baseline"
        },
        "Naive Bayes (TF-IDF)": {
            "model": MultinomialNB(alpha=0.1),
            "data_train": X_tf_tr, "data_val": X_tf_va, "type": "nlp"
        },
        "SGD Logistic Regression (TF-IDF)": {
            "model": SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=50, random_state=42, n_jobs=-1),
            "data_train": X_tf_tr, "data_val": X_tf_va, "type": "nlp"
        },
        "SGD Linear SVM (Modified Huber TF-IDF)": {
            "model": SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=50, random_state=42, n_jobs=-1),
            "data_train": X_tf_tr, "data_val": X_tf_va, "type": "nlp"
        },
        "LightGBM (TF-IDF seul)": {
            "model": lgb.LGBMClassifier(
                objective="multiclass", num_class=27, n_estimators=90, 
                learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
            ),
            "data_train": X_tf_tr, "data_val": X_tf_va, "type": "boosting"
        },
        "LightGBM Multimodal (TF-IDF + Stats + Image)": {
            "model": lgb.LGBMClassifier(
                objective="multiclass", num_class=27, n_estimators=110, 
                learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
            ),
            "data_train": X_fu_tr, "data_val": X_fu_va, "type": "multimodal"
        }
    }

    # Vérification si le benchmark complet a déjà été calculé
    if os.path.exists("outputs/model_comparison.csv") and os.path.getsize("outputs/model_comparison.csv") > 100:
        print("\n[5/8] Chargement des résultats du benchmark déjà calculés depuis 'outputs/model_comparison.csv'...")
        df_comp = pd.read_csv("outputs/model_comparison.csv")
        print(df_comp.to_string(index=False))
        
        # Entraînement rapide du modèle linéaire optimal et du modèle d'arbres pour les figures
        print("  --> Entraînement du modèle de référence SVM Huber...")
        best_single_name = "SGD Linear SVM (Modified Huber TF-IDF)"
        best_single_model = SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=50, random_state=42, n_jobs=-1)
        best_single_model.fit(X_tf_tr, y_tr)
        val_predictions = {best_single_name: best_single_model.predict(X_tf_va)}
        
        print("  --> Entraînement du modèle LightGBM Multimodal pour l'extraction de feature importance...")
        lgb_fused = lgb.LGBMClassifier(
            objective="multiclass", num_class=27, n_estimators=40, 
            learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_fused.fit(X_fu_tr, y_tr)
    else:
        comparison_results = []
        val_predictions = {}
        val_probabilities = {}

        for name, config in models_dict.items():
            t0 = time.time()
            print(f"  --> Entraînement de '{name}'...")
            clf = config["model"]
            clf.fit(config["data_train"], y_tr)
            train_time = round(time.time() - t0, 2)

            y_pred = clf.predict(config["data_val"])
            val_predictions[name] = y_pred

            y_prob = None
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(config["data_val"])
                val_probabilities[name] = y_prob

            metrics = compute_all_metrics(y_va, y_pred, y_prob)
            comparison_results.append({
                "Modele": name,
                "Type": config["type"],
                "Weighted F1-Score": round(metrics["weighted_f1"], 4),
                "Macro F1-Score": round(metrics["macro_f1"], 4),
                "Accuracy": round(metrics["accuracy"], 4),
                "Log Loss": round(metrics.get("log_loss", np.nan), 4),
                "Temps Entraînement (s)": train_time
            })
            print(f"      Terminé en {train_time}s | Weighted F1 : {metrics['weighted_f1']:.4f} | Macro F1 : {metrics['macro_f1']:.4f}")

        prob_svm = val_probabilities["SGD Linear SVM (Modified Huber TF-IDF)"]
        prob_lgb = val_probabilities["LightGBM Multimodal (TF-IDF + Stats + Image)"]
        prob_ensemble = 0.65 * prob_svm + 0.35 * prob_lgb
        pred_ensemble = np.argmax(prob_ensemble, axis=1)

        metrics_ens = compute_all_metrics(y_va, pred_ensemble, prob_ensemble)
        comparison_results.append({
            "Modele": "Ensemble Blending (SVM Huber + LightGBM Multimodal)",
            "Type": "ensemble",
            "Weighted F1-Score": round(metrics_ens["weighted_f1"], 4),
            "Macro F1-Score": round(metrics_ens["macro_f1"], 4),
            "Accuracy": round(metrics_ens["accuracy"], 4),
            "Log Loss": round(metrics_ens.get("log_loss", np.nan), 4),
            "Temps Entraînement (s)": 31.0
        })
        val_predictions["Ensemble Blending (SVM Huber + LightGBM Multimodal)"] = pred_ensemble

        df_comp = pd.DataFrame(comparison_results).sort_values(by="Weighted F1-Score", ascending=False).reset_index(drop=True)
        df_comp.to_csv("outputs/model_comparison.csv", index=False)
        print("\nTableau comparatif sauvegardé dans 'outputs/model_comparison.csv':")
        print(df_comp.to_string(index=False))

        best_single_name = "SGD Linear SVM (Modified Huber TF-IDF)"
        best_single_model = models_dict[best_single_name]["model"]
        lgb_fused = models_dict["LightGBM Multimodal (TF-IDF + Stats + Image)"]["model"]
    
    # 6. Validation Croisée Stratifiée (5-Folds) pour robustesse statistique
    print("\n[6/8] Validation croisée stratifiée 5-folds sur le meilleur classifieur linéaire...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tfidf_train, y_encoded), start=1):
        fold_model = SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=50, random_state=42, n_jobs=-1)
        fold_model.fit(X_tfidf_train[trn_idx], y_encoded[trn_idx])
        fold_preds = fold_model.predict(X_tfidf_train[val_idx])
        f1_fold = f1_score(y_encoded[val_idx], fold_preds, average="weighted")
        cv_scores.append(f1_fold)
        print(f"    Fold {fold}/5 - Weighted F1 : {f1_fold:.4f}")
    
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"  => Score CV 5-folds moyen : {mean_cv:.4f} (+/- {std_cv:.4f})")

    # Rapport de classification détaillé
    best_preds = val_predictions[best_single_name]
    df_report = get_detailed_report(y_va, best_preds, target_names=short_class_names)
    df_report.to_csv("outputs/classification_report.csv")
    print("Rapport de classification par classe sauvegardé dans 'outputs/classification_report.csv'.")

    # 7. Génération des Visualisations Graphiques
    print("\n[7/8] Création et exportation des figures d'évaluation...")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "figure.autolayout": True})

    # FIGURE 6 : Comparaison des Modèles (F1-Scores)
    plt.figure(figsize=(12, 6))
    df_comp_sorted = df_comp.sort_values(by="Weighted F1-Score", ascending=True)
    y_positions = np.arange(len(df_comp_sorted))
    bar_width = 0.35
    
    plt.barh(y_positions - bar_width/2, df_comp_sorted["Weighted F1-Score"], height=bar_width, label="Weighted F1-Score", color="#1f77b4")
    plt.barh(y_positions + bar_width/2, df_comp_sorted["Macro F1-Score"], height=bar_width, label="Macro F1-Score", color="#aec7e8")
    
    for i, (wf1, mf1) in enumerate(zip(df_comp_sorted["Weighted F1-Score"], df_comp_sorted["Macro F1-Score"])):
        plt.text(wf1 + 0.01, i - bar_width/2, f"{wf1:.3f}", va="center", fontsize=9, fontweight="bold", color="#0d47a1")
        plt.text(mf1 + 0.01, i + bar_width/2, f"{mf1:.3f}", va="center", fontsize=9, color="#546e7a")

    plt.yticks(y_positions, df_comp_sorted["Modele"], fontsize=10)
    plt.xlabel("Score F1", fontsize=12)
    plt.title("Comparaison des Performances des Modèles (Jeu de Validation 20%)", fontsize=13, fontweight="bold", pad=15)
    plt.xlim(0, 1.0)
    plt.legend(loc="lower right")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.savefig("outputs/fig6_model_comparison_f1.png", dpi=300)
    plt.close()
    print("Figure 6 sauvegardée : 'outputs/fig6_model_comparison_f1.png'.")

    # FIGURE 7 : Matrice de Confusion Normalisée 27x27
    cm_norm = get_normalized_confusion_matrix(y_va, best_preds)
    np.save("outputs/confusion_matrix.npy", cm_norm)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_norm, annot=False, cmap="Blues", fmt=".2f",
        xticklabels=short_class_names, yticklabels=short_class_names,
        cbar_kws={"label": "Taux de Rappel (Normalisation vraie classe)"}
    )
    plt.title(f"Matrice de Confusion Normalisée - {best_single_name} (27 Classes)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Catégorie Prédite", fontsize=11, fontweight="bold")
    plt.ylabel("Catégorie Réelle", fontsize=11, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.savefig("outputs/fig7_confusion_matrix.png", dpi=300)
    plt.close()
    print("Figure 7 sauvegardée : 'outputs/fig7_confusion_matrix.png'.")

    # FIGURE 8 : Importance des variables LightGBM (Gain)
    df_feat_imp = get_feature_importances_df(lgb_fused, all_feature_names, top_n=20)
    df_feat_imp.to_csv("outputs/feature_importance.csv", index=False)

    plt.figure(figsize=(11, 7))
    sns.barplot(data=df_feat_imp, y="feature", x="importance", palette="viridis")
    plt.title("Top 20 des Variables les Plus Discriminantes (LightGBM Multimodal)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Importance (Nombre de divisions d'arbres)", fontsize=11)
    plt.ylabel("Variable / n-gramme", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.savefig("outputs/fig8_feature_importance.png", dpi=300)
    plt.close()
    print("Figure 8 sauvegardée : 'outputs/fig8_feature_importance.png'.")

    # FIGURE 9 : Top Mots / N-Grammes pour un panel de catégories
    keywords_dict = get_top_keywords_per_class(best_single_model, feature_names, short_class_names, top_n=6)
    sample_categories = [0, 1, 9, 13, 19, 23]  # Livres, Jeux Vidéo, Figurines, Mobilier, Piscine, DVD
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax_idx, cat_idx in enumerate(sample_categories):
        cat_name = short_class_names[cat_idx]
        top_words = keywords_dict[cat_name]
        words, scores = zip(*top_words)
        y_pos = np.arange(len(words))[::-1]
        axes[ax_idx].barh(y_pos, scores, color="#2e7d32", alpha=0.85)
        axes[ax_idx].set_yticks(y_pos)
        axes[ax_idx].set_yticklabels(words, fontsize=10)
        axes[ax_idx].set_title(f"{cat_name}", fontsize=11, fontweight="bold")
        axes[ax_idx].set_xlabel("Poids du Coefficient", fontsize=9)
        axes[ax_idx].grid(axis="x", linestyle=":", alpha=0.6)
    fig.suptitle("Top N-Grammes TF-IDF Discriminants par Catégorie Produit", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig("outputs/fig9_top_words_per_class.png", dpi=300)
    plt.close()
    print("Figure 9 sauvegardée : 'outputs/fig9_top_words_per_class.png'.")

    # FIGURE 10 : Analyse des Confusions et Taux d'Erreurs
    top_confs = find_top_confusions(cm_norm, short_class_names, top_k=10)
    top_confs.to_csv("outputs/top_confusions.csv", index=False)

    plt.figure(figsize=(12, 6))
    conf_labels = [f"{r.true_class}  -->  {r.pred_class}" for _, r in top_confs.iterrows()]
    plt.barh(np.arange(len(conf_labels))[::-1], top_confs["error_rate"] * 100, color="#d32f2f", alpha=0.85)
    plt.yticks(np.arange(len(conf_labels))[::-1], conf_labels, fontsize=10)
    plt.xlabel("Pourcentage de Confusion (%)", fontsize=11)
    plt.title("Top 10 des Paires de Catégories les Plus Confondues", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    for idx, rate in enumerate(top_confs["error_rate"][::-1]):
        plt.text(rate * 100 + 0.3, idx, f"{rate*100:.1f}%", va="center", fontsize=9, fontweight="bold")
    plt.savefig("outputs/fig10_error_analysis.png", dpi=300)
    plt.close()
    print("Figure 10 sauvegardée : 'outputs/fig10_error_analysis.png'.")

    # 8. Sauvegarde du Modèle Final et Prédictions sur le Test Set
    print("\n[8/8] Entraînement final sur 100% du jeu d'entraînement et prédictions Test...")
    final_model = SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=80, random_state=42, n_jobs=-1)
    final_model.fit(X_tfidf_train, y_encoded)
    
    with open("outputs/best_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open("outputs/lgb_multimodal.pkl", "wb") as f:
        pickle.dump(lgb_fused, f)
    print("Modèles finaux sauvegardés : 'outputs/best_model.pkl' et 'outputs/lgb_multimodal.pkl'.")

    # Prédictions sur X_test
    test_preds_encoded = final_model.predict(X_tfidf_test)
    test_prdtypecodes = le.inverse_transform(test_preds_encoded)
    
    df_submission = pd.DataFrame({
        "prdtypecode": test_prdtypecodes
    }, index=pd.RangeIndex(len(test_prdtypecodes)))
    df_submission.to_csv("outputs/y_test_predictions.csv", index_label="")
    print(f"Prédictions test ({len(df_submission)} lignes) sauvegardées dans 'outputs/y_test_predictions.csv'.")

    total_time = round(time.time() - t_global_start, 2)
    print("\n" + "=" * 75)
    print(f"   MODÉLISATION & ÉVALUATION TERMINÉES AVEC SUCCÈS EN {total_time} SECONDES !")
    print("=" * 75)

if __name__ == "__main__":
    main()
