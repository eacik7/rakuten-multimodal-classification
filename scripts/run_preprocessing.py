import os
import sys
import pickle
import time
import pandas as pd
from scipy.sparse import save_npz

# Ajout du dossier racine au sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_raw_data, PRODUCT_CODE_MAP
from src.preprocessing.text_cleaner import preprocess_text_dataframe
from src.preprocessing.feature_engineering import extract_text_stats_features, create_tfidf_features
from src.preprocessing.image_processor import extract_images_zip, extract_image_features_dataframe

def main():
    print("=" * 70)
    print("      RAKUTEN MULTIMODAL - ÉTAPE 2 : PRE-PROCESSING & FEATURE ENGINEERING")
    print("=" * 70)
    start_time = time.time()

    # 1. Chargement des données brutes
    print("\n[1/5] Chargement des jeux de données bruts...")
    df_x_train, df_y_train, df_x_test = load_raw_data(".")
    print(f"Train samples: {len(df_x_train)} | Test samples: {len(df_x_test)}")

    # 2. Décompression des images si nécessaire
    print("\n[2/5] Vérification et décompression des images...")
    extract_images_zip(zip_path="images.zip", target_dir="images")

    # 3. Pre-processing Texte
    print("\n[3/5] Application du pré-traitement textuel (HTML, Unicode, Stopwords)...")
    df_x_train_clean = preprocess_text_dataframe(df_x_train)
    df_x_test_clean = preprocess_text_dataframe(df_x_test)

    # Sauvegarde des CSV nettoyés
    df_x_train_clean.to_csv("X_train_clean.csv")
    df_x_test_clean.to_csv("X_test_clean.csv")
    print("CSV nettoyés sauvegardés : 'X_train_clean.csv' et 'X_test_clean.csv'.")

    # 4. Feature Engineering Texte & Statistiques
    print("\n[4/5] Inscription des nouvelles variables (Text Stats & TF-IDF)...")
    train_stats = extract_text_stats_features(df_x_train_clean)
    test_stats = extract_text_stats_features(df_x_test_clean)

    # Extraction TF-IDF (5000 features)
    tfidf_train, tfidf_test, vectorizer = create_tfidf_features(
        train_texts=df_x_train_clean["text_full"],
        test_texts=df_x_test_clean["text_full"],
        max_features=5000,
        ngram_range=(1, 2)
    )

    # Sauvegarde des features tabulaires et matrices TF-IDF
    os.makedirs("outputs", exist_ok=True)
    train_stats.to_csv("outputs/train_features.csv")
    test_stats.to_csv("outputs/test_features.csv")

    save_npz("outputs/tfidf_train.npz", tfidf_train)
    save_npz("outputs/tfidf_test.npz", tfidf_test)
    with open("outputs/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Features statistiques et matrices TF-IDF sauvegardées dans 'outputs/'.")

    # 5. Extraction des métadonnées d'images
    print("\n[5/5] Extraction des propriétés et métadonnées d'images...")
    # On extrait un échantillon rapide ou la totalité selon présence des dossiers images
    if os.path.exists("images"):
        img_meta_train = extract_image_features_dataframe(df_x_train, base_dir="images")
        img_meta_test = extract_image_features_dataframe(df_x_test, base_dir="images")
        img_meta_train.to_csv("outputs/image_meta_train.csv")
        img_meta_test.to_csv("outputs/image_meta_test.csv")
        print("Métadonnées d'images sauvegardées dans 'outputs/image_meta_train.csv'.")

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"   PRE-PROCESSING ET FEATURE ENGINEERING TERMINÉS EN {elapsed} SECONDES !")
    print("=" * 70)

if __name__ == "__main__":
    main()
