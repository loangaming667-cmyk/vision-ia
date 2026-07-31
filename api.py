"""
api.py
------
Serveur local (FastAPI) qui expose le modèle de détection au Front-End.

Ce fichier fait 2 choses :
1) Il sert l'interface (interface.html) quand on ouvre http://localhost:8000/
2) Il expose une route /detect qui reçoit une image et renvoie les
   objets détectés en JSON.

Les deux sont sur le MÊME serveur (même port 8000) exprès : ça évite
les soucis de "CORS" (le navigateur qui bloque les requêtes entre deux
origines différentes). Un seul serveur = un seul port = pas de problème.

Rôle dans le projet : Membre 3 - Développeur Back-End
Lancement : uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import io

from inference import detect_objects, CLASS_NAMES

# ---------------------------------------------------------------
# Création de l'application FastAPI
# ---------------------------------------------------------------
app = FastAPI(
    title="API de reconnaissance d'objets",
    description="Reçoit une image, renvoie les objets détectés par le modèle YOLOv8 du Membre 2.",
    version="1.0.0",
)


@app.get("/")
def serve_interface():
    """
    Sert le fichier interface.html quand on visite http://localhost:8000/
    C'est ce qui permet d'avoir Front-End + Back-End sur le même serveur.
    """
    return FileResponse("interface.html")


@app.get("/health")
def health_check():
    """
    Petite route de vérification : permet de tester rapidement si le
    serveur et le modèle sont bien chargés, sans envoyer d'image.
    Pratique pour débugger : http://localhost:8000/health
    """
    return {
        "status": "ok",
        "model_loaded": True,
        "num_classes": len(CLASS_NAMES),
        "classes": list(CLASS_NAMES.values()),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Route principale : reçoit une image envoyée par le Front-End
    (bouton "Scanner") et renvoie la liste des objets détectés.

    Le Front-End envoie l'image en "multipart/form-data" (comme un
    formulaire d'upload de fichier classique), c'est le format standard
    et le plus simple à gérer côté navigateur avec fetch() + FormData.

    Réponse JSON renvoyée :
    {
        "detections": [
            {"label": "Laptop", "confidence": 0.91,
             "box": {"x1":..., "y1":..., "x2":..., "y2":...}},
            ...
        ],
        "image_width": 1280,
        "image_height": 720
    }
    """
    # Vérification basique : on n'accepte que des images.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas une image.")

    try:
        # On lit les octets bruts envoyés par le navigateur...
        image_bytes = await file.read()
        # ...et on les transforme en image PIL exploitable par le modèle.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image envoyée.")

    detections = detect_objects(image)

    return JSONResponse({
        "detections": detections,
        "image_width": image.width,
        "image_height": image.height,
    })
