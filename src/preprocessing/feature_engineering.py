import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_text_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des variables statistiques et métadonnées textuelles :
    - Longueurs en caractères et mots (désignation, description, text_full).
    - Ratios de majuscules et de chiffres.
    - Indicateur de présence de description.
    """
    df_feat = pd.DataFrame(index=df.index)

    # Textes bruts
    raw_desig = df["designation"].fillna("").astype(str)
    raw_desc = df["description"].fillna("").astype(str)

    # Indicateur binaire de description
    df_feat["has_description"] = (raw_desc.str.strip().str.len() > 0).astype(int)

    # Longueurs désignation
    df_feat["desig_char_len"] = raw_desig.str.len()
    df_feat["desig_word_count"] = raw_desig.apply(lambda x: len(x.split()))

    # Longueurs description
    df_feat["desc_char_len"] = raw_desc.str.len()
    df_feat["desc_word_count"] = raw_desc.apply(lambda x: len(x.split()))

    # Compte et ratios de majuscules (désignation)
    desig_caps = raw_desig.apply(lambda x: sum(1 for c in x if c.isupper()))
    df_feat["desig_caps_count"] = desig_caps
    df_feat["desig_caps_ratio"] = (desig_caps / (df_feat["desig_char_len"] + 1e-5)).round(4)

    # Compte et ratios de chiffres (désignation & description)
    desig_digits = raw_desig.apply(lambda x: sum(1 for c in x if c.isdigit()))
    df_feat["desig_digit_count"] = desig_digits
    df_feat["desig_digit_ratio"] = (desig_digits / (df_feat["desig_char_len"] + 1e-5)).round(4)

    desc_digits = raw_desc.apply(lambda x: sum(1 for c in x if c.isdigit()))
    df_feat["desc_digit_count"] = desc_digits
    df_feat["desc_digit_ratio"] = (desc_digits / (df_feat["desc_char_len"] + 1e-5)).round(4)

    # Textes nettoyés si présents
    if "designation_clean" in df.columns:
        df_feat["desig_clean_word_count"] = df["designation_clean"].apply(lambda x: len(str(x).split()))
    if "description_clean" in df.columns:
        df_feat["desc_clean_word_count"] = df["description_clean"].apply(lambda x: len(str(x).split()))
    if "text_full" in df.columns:
        df_feat["text_full_word_count"] = df["text_full"].apply(lambda x: len(str(x).split()))

    return df_feat

def create_tfidf_features(train_texts: pd.Series, test_texts: pd.Series = None, max_features: int = 5000, ngram_range: tuple = (1, 2)):
    """
    Applique la vectorisation TF-IDF sur le texte fusionné.
    Returns:
        tfidf_train (scipy.sparse matrix), tfidf_test, vectorizer
    """
    print(f"Extraction TF-IDF (max_features={max_features}, ngrams={ngram_range})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=3,
        sublinear_tf=True
    )
    
    tfidf_train = vectorizer.fit_transform(train_texts.fillna(""))
    tfidf_test = vectorizer.transform(test_texts.fillna("")) if test_texts is not None else None

    return tfidf_train, tfidf_test, vectorizer

if __name__ == "__main__":
    df_dummy = pd.DataFrame({
        "designation": ["Olivia: Personalisiertes Notizbuch 150", "Journal Des Arts N 133"],
        "description": ["Notizbuch fur Kinder", None]
    })
    stats = extract_text_stats_features(df_dummy)
    print(stats)
