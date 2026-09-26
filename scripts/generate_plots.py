import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data_loader import load_raw_data

def generate_plots():
    os.makedirs("outputs", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    print("Chargement des données pour la génération des graphiques...")
    df_x_train, df_y_train, _ = load_raw_data(".")
    df = pd.concat([df_x_train, df_y_train], axis=1)

    # Figure 1 : Distribution des Classes Cibles
    plt.figure(figsize=(12, 7))
    class_counts = df["category_name"].value_counts()
    ax = sns.barplot(x=class_counts.values, y=class_counts.index, palette="mako")
    plt.title("Distribution des 27 Catégories de Produits (Rakuten Train Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Nombre d'Annonces")
    plt.ylabel("Catégorie de Produit")
    for p in ax.patches:
        width = p.get_width()
        pct = (width / len(df)) * 100
        ax.annotate(f"{int(width):,} ({pct:.1f}%)",
                    (width + 100, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/fig1_class_distribution.png", dpi=300)
    plt.close()
    print("Généré: outputs/fig1_class_distribution.png")

    # Figure 2 : Taux d'Absence de Description par Catégorie
    plt.figure(figsize=(12, 7))
    df["has_desc"] = df["description"].notna()
    missing_by_class = df.groupby("category_name")["has_desc"].apply(lambda x: (1 - x.mean()) * 100).sort_values(ascending=False)
    ax2 = sns.barplot(x=missing_by_class.values, y=missing_by_class.index, palette="rocket")
    plt.title("Taux d'Absence de Description (%) par Catégorie", fontsize=14, fontweight="bold")
    plt.xlabel("Pourcentage de Descriptions Manquantes (%)")
    plt.ylabel("Catégorie")
    for p in ax2.patches:
        width = p.get_width()
        ax2.annotate(f"{width:.1f}%",
                     (width + 1, p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/fig2_missing_description_by_class.png", dpi=300)
    plt.close()
    print("Généré: outputs/fig2_missing_description_by_class.png")

    # Figure 3 : Longueur Moyenne de Désignation par Catégorie
    plt.figure(figsize=(12, 7))
    df["desig_len"] = df["designation"].fillna("").str.len()
    avg_len = df.groupby("category_name")["desig_len"].mean().sort_values(ascending=False)
    ax3 = sns.barplot(x=avg_len.values, y=avg_len.index, palette="crest")
    plt.title("Longueur Moyenne des Titres (Caractères) par Catégorie", fontsize=14, fontweight="bold")
    plt.xlabel("Longueur Moyenne (Nb de caractères)")
    plt.ylabel("Catégorie")
    for p in ax3.patches:
        width = p.get_width()
        ax3.annotate(f"{width:.1f}",
                     (width + 1, p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/fig3_title_length_by_class.png", dpi=300)
    plt.close()
    print("Généré: outputs/fig3_title_length_by_class.png")

    print("Tous les graphiques ont été générés dans 'outputs/'.")

if __name__ == "__main__":
    generate_plots()
