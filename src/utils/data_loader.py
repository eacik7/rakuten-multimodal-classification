import os
import pandas as pd

# Mapping des 27 codes de produits (prdtypecode) vers leurs catégories explicites
PRODUCT_CODE_MAP = {
    10: "Livres / Romans / Brochés",
    40: "Jeux Vidéo (Jeux)",
    50: "Accessoires Gaming & Manettes",
    60: "Consoles de Jeux Vidéo",
    1140: "Figurines & Produits dérivés",
    1160: "Cartes à collectionner",
    1180: "Figurines Wargame & Jeux de rôle",
    1280: "Jouets, Peluches & Poupées",
    1281: "Jeux de Société & Puzzles",
    1300: "Modélisme, Drones & Véhicules RC",
    1301: "Jeux de plein air & Enfants",
    1302: "Jeux d'imitation, Dînette & Déguisements",
    1320: "Puériculture & Équipement Bébé",
    1560: "Mobilier & Ameublement Intérieur",
    1920: "Linge de Maison & Literie",
    1940: "Épicerie, Alimentation & Boissons",
    2060: "Décoration, Éclairage & Jardin",
    2220: "Animalerie & Accessoires Animaux",
    2280: "Presse, Journaux & Magazines",
    2403: "Livres Enfants, BD & Mangas",
    2462: "Consoles & Packs Consoles",
    2522: "Papeterie & Fournitures Bureau",
    2582: "Jardinage & Mobilier Extérieur",
    2583: "Piscine, Spa & Entretien",
    2585: "Bricolage, Outillage & Électricité",
    2705: "Films, DVD & Blu-Ray",
    2905: "Jeux Vidéo Dématérialisés (DLC/Codes)",
}

def get_category_name(code: int) -> str:
    """Retourne le nom explicite de la catégorie à partir du prdtypecode."""
    return PRODUCT_CODE_MAP.get(int(code), f"Inconnu ({code})")

def load_raw_data(data_dir: str = "."):
    """
    Charge les fichiers CSV bruts X_train, Y_train et X_test.
    Returns:
        df_x_train, df_y_train, df_x_test
    """
    x_train_path = os.path.join(data_dir, "X_train_update.csv")
    y_train_path = os.path.join(data_dir, "Y_train_CVw08PX.csv")
    x_test_path = os.path.join(data_dir, "X_test_update.csv")
    
    df_x_train = pd.read_csv(x_train_path, index_col=0)
    df_y_train = pd.read_csv(y_train_path, index_col=0)
    df_x_test = pd.read_csv(x_test_path, index_col=0)
    
    # Ajout des libellés de catégories explicites dans df_y_train
    df_y_train["category_name"] = df_y_train["prdtypecode"].map(PRODUCT_CODE_MAP)
    
    return df_x_train, df_y_train, df_x_test

if __name__ == "__main__":
    x_tr, y_tr, x_te = load_raw_data()
    print(f"X_train shape: {x_tr.shape}")
    print(f"Y_train shape: {y_tr.shape}")
    print(f"X_test shape:  {x_te.shape}")
    print("Exemple Y_train:")
    print(y_tr.head())
