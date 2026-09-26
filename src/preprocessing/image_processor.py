import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def extract_images_zip(zip_path: str = "images.zip", target_dir: str = "images") -> bool:
    """
    Décompresse l'archive images.zip vers le dossier cible si nécessaire.
    """
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"Le dossier '{target_dir}' existe déjà et n'est pas vide.")
        return True

    if not os.path.exists(zip_path):
        print(f"Fichier d'archive '{zip_path}' introuvable.")
        return False

    print(f"Extraction de '{zip_path}' vers '{target_dir}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction terminée avec succès.")
    return True

def get_image_path(image_id: int, product_id: int, base_dir: str = "images") -> str:
    """
    Recherche l'image correspondant à image_id et product_id dans image_train ou image_test.
    """
    filename = f"image_{image_id}_product_{product_id}.jpg"
    train_path = os.path.join(base_dir, "image_train", filename)
    test_path = os.path.join(base_dir, "image_test", filename)

    if os.path.exists(train_path):
        return train_path
    elif os.path.exists(test_path):
        return test_path
    return ""

def extract_single_image_features(img_path: str) -> dict:
    """
    Extrait les propriétés statistiques d'une image JPG :
    - Intégrité / validité
    - Dimensions (largeur, hauteur, aspect ratio)
    - Statistiques des canaux RGB (moyennes, écarts-types)
    - Luminosité moyenne & contraste (écart-type du niveau de gris)
    """
    default_stats = {
        "img_exists": False,
        "img_width": 0,
        "img_height": 0,
        "img_aspect_ratio": 0.0,
        "img_mean_r": 0.0,
        "img_mean_g": 0.0,
        "img_mean_b": 0.0,
        "img_std_r": 0.0,
        "img_std_g": 0.0,
        "img_std_b": 0.0,
        "img_brightness": 0.0,
        "img_contrast": 0.0
    }

    if not img_path or not os.path.exists(img_path):
        return default_stats

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            arr = np.array(img, dtype=np.float32)

            mean_rgb = arr.mean(axis=(0, 1))
            std_rgb = arr.std(axis=(0, 1))

            gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
            brightness = float(gray.mean())
            contrast = float(gray.std())

            return {
                "img_exists": True,
                "img_width": w,
                "img_height": h,
                "img_aspect_ratio": round(w / h, 4) if h > 0 else 1.0,
                "img_mean_r": round(float(mean_rgb[0]), 2),
                "img_mean_g": round(float(mean_rgb[1]), 2),
                "img_mean_b": round(float(mean_rgb[2]), 2),
                "img_std_r": round(float(std_rgb[0]), 2),
                "img_std_g": round(float(std_rgb[1]), 2),
                "img_std_b": round(float(std_rgb[2]), 2),
                "img_brightness": round(brightness, 2),
                "img_contrast": round(contrast, 2)
            }
    except Exception as e:
        print(f"Erreur de lecture image {img_path}: {e}")
        return default_stats

def extract_image_features_dataframe(df: pd.DataFrame, base_dir: str = "images", max_samples: int = None) -> pd.DataFrame:
    """
    Extrait les propriétés d'images pour un DataFrame complet.
    """
    records = []
    target_df = df if max_samples is None else df.iloc[:max_samples]

    print(f"Extraction des caractéristiques images sur {len(target_df)} lignes...")
    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        img_id = row["imageid"]
        prod_id = row["productid"]
        img_path = get_image_path(img_id, prod_id, base_dir=base_dir)
        stats = extract_single_image_features(img_path)
        stats["productid"] = prod_id
        stats["imageid"] = img_id
        stats["image_path"] = img_path
        records.append(stats)

    res_df = pd.DataFrame(records)
    res_df.set_index(target_df.index, inplace=True)
    return res_df

if __name__ == "__main__":
    extract_images_zip()
