import openpyxl

def update_template():
    excel_path = "Template - Rapport exploration des données.xlsx"
    wb = openpyxl.load_workbook(excel_path)
    sheet = wb["Template"]

    # Nouvelles colonnes / variables créées lors de l'Étape 2
    new_rows = [
        (6, "designation_clean", "Titre du produit nettoyé (sans HTML, Unicode, sans stop-words)", "Oui. Dérivé de designation.", "object (String / Texte nettoyé)", "0.00 %", "Prétraitement NLP complet (unicodedata, lowercase, stopwords FR/EN).", "Longueur moyenne : ~8.2 mots.", "Champ de base pour le TF-IDF et les modèles NLP (CamemBERT)."),
        (7, "description_clean", "Description nettoyée sans balises HTML ni entités", "Oui. Dérivée de description.", "object (String / Texte nettoyé)", "35.09 %", "NaN remplacés par une chaîne vide ''", "Longueur moyenne quand présente : ~95 mots.", "Nettoyé par Regex (<p>, <br>, CSS style) et décodage html.unescape."),
        (8, "text_full", "Texte fusionné (designation_clean + description_clean)", "Oui.", "object (String / Texte fusionné)", "0.00 %", "Aucune valeur manquante (garanti).", "Concaténation complète sans rupture d'information.", "Fournit le champ d'entrée principal pour la vectorisation TF-IDF (5 000 dimensions)."),
        (9, "has_description", "Indicateur binaire de présence de description", "Oui.", "int64 (Binaire 0/1)", "0.00 %", "Aucune action nécessaire (0 ou 1).", "Train : 64.91 % à 1 (présente), 35.09 % à 0 (absente).", "Variable statistique hautement discriminante par catégorie (ex: presse vs mobilier)."),
        (10, "desig_caps_ratio", "Proportion de lettres majuscules dans le titre brut", "Oui.", "float64 (Ratio [0.0, 1.0])", "0.00 %", "Aucune action nécessaire.", "Moyenne : ~0.15. Pics élevés sur consoles et jeux vidéo.", "Distingue les références/marques (ex: PS4, XBOX, DVD)."),
        (11, "desig_digit_ratio", "Proportion de chiffres dans le titre brut", "Oui.", "float64 (Ratio [0.0, 1.0])", "0.00 %", "Aucune action en cas de NA.", "Moyenne : ~0.08. Utile pour tomes, volumes, années.", "Capture les numéros de séries et tomes de mangas/magazines."),
        (12, "img_brightness", "Luminosité moyenne de l'image (0 à 255)", "Oui.", "float64 (Statistique image)", "0.00 %", "Extraction depuis fichier JPG via NumPy/PIL.", "Moyenne global : ~210.0 (fonds studio blancs).", "Indicateur visuel utile pour la modération et la détection d'arrière-plans.")
    ]

    start_row = 9
    for r_idx, data in enumerate(new_rows, start=start_row):
        for c_idx, val in enumerate(data, start=1):
            sheet.cell(row=r_idx, column=c_idx, value=val)

    wb.save(excel_path)
    print(f"Modifications enregistrées dans '{excel_path}'.")

if __name__ == "__main__":
    update_template()
