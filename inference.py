"""
inference.py
------------
Ce module contient toute la "logique métier" de détection d'objets.
Il charge le modèle YOLOv8 (last.pt) une seule fois au démarrage,
puis expose une fonction detect_objects() que l'API (api.py) appelle
à chaque photo envoyée par le Front-End.

Rôle dans le projet : Membre 3 - Développeur Back-End
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------------------------------------------------------
# 1) Chargement du modèle
# ---------------------------------------------------------------
# On charge le modèle UNE SEULE FOIS quand le serveur démarre,
# car charger un modèle YOLO à chaque requête serait très lent
# (plusieurs secondes) et rendrait l'API inutilisable en pratique.
MODEL_PATH = "last.pt"
model = YOLO(MODEL_PATH)

# Dictionnaire {id: nom_classe} récupéré directement depuis le modèle
# entraîné par le Membre 2 (ex : {0: "Backpack", 1: "Bottle_1", ...})
CLASS_NAMES = model.names

# Le modèle a été entraîné avec des noms de classes en anglais.
# On garde ces noms tels quels en interne (ce sont les vrais noms des
# classes, utiles pour le débogage), mais on les traduit pour l'affichage
# côté utilisateur. Si une classe n'est pas dans ce dictionnaire, on
# affichera son nom anglais original (voir translate_label ci-dessous).
FRENCH_LABELS = {
    "Backpack": "Sac à dos",
    "Bottle_1": "Bouteille (type 1)",
    "Bottle_2": "Bouteille (type 2)",
    "Bottle_3": "Bouteille (type 3)",
    "Bottle_4": "Bouteille (type 4)",
    "Bottle_5": "Bouteille (type 5)",
    "Bottle_6": "Bouteille (type 6)",
    "Boots": "Bottes",
    "Flip_flops": "Tongs",
    "Loafers": "Mocassins",
    "Sandals": "Sandales",
    "Sneakers": "Baskets",
    "Soccer": "Ballon de foot",
    "Laptop": "Ordinateur portable",
    "Smartphone": "Smartphone",
    "Tablet": "Tablette",
    "Pen": "Stylo",
    "Chair": "Chaise",
    "Table": "Table",
}


def translate_label(english_label: str) -> str:
    """Traduit un nom de classe anglais en français. Si la classe n'est
    pas dans le dictionnaire (ex : ajoutée plus tard par le Membre 2 sans
    mise à jour ici), on renvoie le nom anglais tel quel plutôt que de planter."""
    return FRENCH_LABELS.get(english_label, english_label)


# Seuil de confiance minimum : en dessous, on considère que la détection
# n'est pas assez fiable et on l'ignore (évite le "bruit").
# 0.25 est un bon compromis "détecte plus, quitte à se tromper un peu plus" ;
# monte-le (ex 0.5) si tu préfères moins de détections mais plus fiables.
CONFIDENCE_THRESHOLD = 0.25


def detect_objects(image: Image.Image, conf_threshold: float = CONFIDENCE_THRESHOLD) -> list[dict]:
    """
    Prend une image (format PIL) et retourne la liste des objets détectés.

    Paramètres
    ----------
    image : PIL.Image
        L'image sur laquelle on veut détecter des objets.
    conf_threshold : float
        Seuil de confiance minimum (entre 0 et 1) pour garder une détection.

    Retour
    ------
    list[dict] : une liste de dictionnaires, un par objet détecté, ex :
        [
            {
                "label": "Laptop",
                "confidence": 0.91,
                "box": {"x1": 120, "y1": 45, "x2": 430, "y2": 300}
            },
            ...
        ]
        Les coordonnées de la boîte sont en PIXELS, dans le repère de
        l'image d'origine envoyée (pas encore mises à l'échelle pour
        l'écran : ça, c'est le Front-End qui s'en charge).
    """
    # Ultralytics accepte directement une image PIL ou un tableau numpy.
    # verbose=False : évite d'afficher plein de logs dans le terminal
    # à chaque requête.
    results = model.predict(source=image, conf=conf_threshold, verbose=False)

    detections = []
    # results est une liste (une entrée par image envoyée). Ici on n'en
    # envoie qu'une seule à la fois, donc on prend results[0].
    result = results[0]

    for box in result.boxes:
        class_id = int(box.cls[0])                  # index numérique de la classe
        english_label = CLASS_NAMES[class_id]         # nom d'origine du modèle, ex "Laptop"
        confidence = float(box.conf[0])               # score de confiance (0 à 1)
        x1, y1, x2, y2 = box.xyxy[0].tolist()          # coordonnées du rectangle

        detections.append({
            "label": translate_label(english_label),  # nom affiché à l'utilisateur (français)
            "label_en": english_label,                 # nom d'origine, gardé au cas où (debug, logs, export...)
            "confidence": round(confidence, 4),
            "box": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            },
        })

    # On trie du plus confiant au moins confiant : plus agréable à afficher.
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ---------------------------------------------------------------
# 2) Mode "test en ligne de commande"
# ---------------------------------------------------------------
# Ceci permet de tester le script TOUT SEUL, sans lancer l'API,
# juste pour vérifier que le modèle fonctionne bien.
# Utilisation : python inference.py chemin/vers/une_photo.jpg
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python inference.py chemin/vers/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    img = Image.open(image_path).convert("RGB")
    detections = detect_objects(img)

    print(f"\n{len(detections)} objet(s) détecté(s) dans {image_path} :\n")
    for d in detections:
        print(f"  - {d['label']} (confiance : {d['confidence']*100:.1f}%) -> boîte {d['box']}")
