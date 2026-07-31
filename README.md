# Back-End — Reconnaissance d'objets (Membre 3)

## Ce que contient ce dossier

| Fichier            | Rôle                                                                 |
|--------------------|-----------------------------------------------------------------------|
| `inference.py`     | Charge `last.pt` (modèle YOLOv8 du Membre 2) et détecte les objets d'une image. |
| `api.py`           | Serveur FastAPI : sert l'interface ET expose la route `/detect`.       |
| `interface.html`   | Interface du Membre 1, modifiée pour appeler l'API réelle.           |
| `requirements.txt` | Liste des librairies Python à installer.                             |
| `last.pt`          | Le modèle entraîné (fourni par le Membre 2).                          |

## Comment lancer le projet (tout en local)

1. **Installer les dépendances** (une seule fois) :
   ```bash
   pip install -r requirements.txt
   ```

2. **Lancer le serveur** :
   ```bash
   uvicorn api:app --reload --port 8000
   ```
   Le `--reload` recharge automatiquement le serveur si tu modifies le code (pratique en développement).

3. **Ouvrir l'application** dans le navigateur :
   ```
   http://localhost:8000/
   ```
   ⚠️ Important : ouvre bien cette URL (pas le fichier `interface.html` en double-clic), sinon la caméra et l'API ne fonctionneront pas correctement.

4. Clique sur **Scanner** dans la barre du bas, autorise la caméra, vise un objet, et appuie sur le bouton rond pour lancer une détection.

## Comment ça marche (pour ta soutenance / rapport)

1. Le navigateur capture une image de la caméra au moment du clic sur "Scanner".
2. Cette image est envoyée en `POST` (format `multipart/form-data`) vers `/detect`.
3. `api.py` reçoit l'image, la transmet à `inference.py`.
4. `inference.py` fait tourner le modèle YOLOv8 dessus et renvoie la liste des objets trouvés (nom, confiance, position).
5. `api.py` renvoie cette liste au navigateur au format JSON.
6. Le JavaScript de `interface.html` affiche les résultats dans la liste ET dessine les boîtes de détection sur l'image.

## Pourquoi le Front-End et le Back-End sont sur le même serveur

Normalement, quand une page web (une "origine") appelle une API sur un port ou domaine différent, le navigateur peut bloquer la requête par sécurité (erreur "CORS"). En servant `interface.html` directement depuis FastAPI (route `/`), tout se passe sur `http://localhost:8000`, donc ce problème n'existe pas. C'est la solution la plus simple et la plus fiable pour un projet local.

## Tester l'API seule (sans navigateur), utile pour débugger

```bash
# Vérifier que le serveur et le modèle sont bien chargés
curl http://localhost:8000/health

# Tester une détection sur une image précise
curl -X POST http://localhost:8000/detect -F "file=@chemin/vers/photo.jpg"
```

Ou en ligne de commande directement, sans même passer par l'API :
```bash
python inference.py chemin/vers/photo.jpg
```

## Classes que le modèle sait reconnaître

Backpack, Bottle_1 à Bottle_6, Boots, Flip_flops, Loafers, Sandals, Sneakers, Soccer, Laptop, Smartphone, Tablet, Pen, Chair, Table.

## Limites connues / pistes d'amélioration (à mentionner dans le rapport)

- Le scan se fait sur une **photo unique** au clic (et non en flux continu), pour rester simple et fiable en local — c'était le choix demandé pour ce rendu.
- Le seuil de confiance est fixé à `0.35` dans `inference.py` (variable `CONFIDENCE_THRESHOLD`) : tu peux l'ajuster si le modèle détecte trop peu, ou trop, d'objets.
- Pas d'authentification sur l'API : normal pour un projet local/étudiant, à ne pas exposer tel quel sur internet.
