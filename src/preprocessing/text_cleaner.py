import re
import html
import sys
import pandas as pd

try:
    import nltk
    from nltk.stem.snowball import FrenchStemmer
    STEMMER = FrenchStemmer()
except ImportError:
    STEMMER = None

# French & English stop words fallback list
FRENCH_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux",
    "il", "ils", "je", "la", "le", "les", "leur", "lui", "ma", "mais", "me", "même", "mes", "moi",
    "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui",
    "sa", "se", "ses", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos",
    "votre", "vous", "c", "d", "j", "l", "à", "m", "n", "s", "t", "y", "été", "étée", "étés",
    "étées", "étant", "suis", "es", "est", "sommes", "êtes", "sont", "serai", "seras", "sera",
    "serons", "serez", "seront", "serais", "serait", "serions", "seriez", "seraient", "étais",
    "était", "étions", "étiez", "étaient", "fus", "fut", "fûmes", "fûtes", "furent", "sois",
    "soit", "soyons", "soyez", "soient", "fusse", "fusses", "fût", "fussions", "fussiez", "fussent",
    "ayant", "eu", "eue", "eues", "eus", "ai", "as", "avons", "avez", "ont", "aurai", "auras",
    "aura", "aurons", "aurez", "auront", "aurais", "aurait", "aurions", "auriez", "auraient",
    "avais", "avait", "avions", "aviez", "avaient", "eut", "eûmes", "eûtes", "eurent", "aie",
    "aies", "ait", "ayons", "ayez", "aient", "eusse", "eusses", "eût", "eussions", "eussiez", "eussent"
}

ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}

ALL_STOPWORDS = FRENCH_STOPWORDS.union(ENGLISH_STOPWORDS)

def clean_raw_text(text: str, remove_stopwords: bool = True, stem: bool = False) -> str:
    """
    Nettoie le texte brut de Rakuten :
    1. Traitement des valeurs nuls / non-chaînes.
    2. Décodage des entités HTML (&eacute;, &amp;, etc.).
    3. Suppression des balises HTML (<p>, <br>, CSS style, etc.).
    4. Suppression du caractère de remplacement Unicode chr(65533).
    5. Normalisation minuscules et ponctuation.
    6. (Optionnel) Suppression des stop-words et racinisation.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""

    # 1. HTML Unescape
    text = html.unescape(text)

    # 2. Suppression des balises HTML / CSS / JS
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Remplacement des caractères de remplacement ou de contrôle
    text = text.replace(chr(65533), " ")
    text = re.sub(r"[\r\n\t]+", " ", text)

    # 4. Passage en minuscules
    text = text.lower()

    # 5. Nettoyage de la ponctuation en conservant les lettres Unicode et chiffres
    text = re.sub(r"[^\w\s]", " ", text)

    # 6. Découpage en mots
    words = text.split()

    # 7. Suppression des stopwords
    if remove_stopwords:
        words = [w for w in words if w not in ALL_STOPWORDS and len(w) > 1]

    # 8. Racinisation optionnelle
    if stem and STEMMER is not None:
        words = [STEMMER.stem(w) for w in words]

    return " ".join(words)

def preprocess_text_dataframe(df: pd.DataFrame, stem: bool = False) -> pd.DataFrame:
    """
    Applique le nettoyage texte complet sur le DataFrame.
    Génère :
    - designation_clean
    - description_clean
    - text_full (désignation + description nettoyées)
    """
    df = df.copy()
    
    # Remplacement des NaN dans description par chaîne vide
    df["description"] = df["description"].fillna("")

    print("Nettoyage des désignations...")
    df["designation_clean"] = df["designation"].apply(lambda x: clean_raw_text(x, remove_stopwords=True, stem=stem))

    print("Nettoyage des descriptions...")
    df["description_clean"] = df["description"].apply(lambda x: clean_raw_text(x, remove_stopwords=True, stem=stem))

    # Fusion désignation + description
    df["text_full"] = (df["designation_clean"] + " " + df["description_clean"]).str.strip()

    return df

if __name__ == "__main__":
    sample = "<p>Journal Des Arts (Le) N 133 Du 28/09/2001 - L'art Et Son Ma&eacute;cenat</p>"
    print("Avant :", sample)
    print("Après :", clean_raw_text(sample))
