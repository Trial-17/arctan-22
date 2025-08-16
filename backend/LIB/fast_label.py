import json
from pathlib import Path
import base64
import os
import av
from PIL import Image
from pymediainfo import MediaInfo
import requests

# Supposons qu'un fichier de configuration existe pour l'URL de l'API et le statut
from LIB import config

# --- Configuration de la structure de l'application ---

def create_fast_app_structure():
    """
    Crée une structure de dossiers dédiée pour le label rapide.
    """
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "FastLabel"
    folders = {
        "db": str(base_path / "db"),
        "thumbnails": str(base_path / "thumbnails"),
    }
    for folder_path in folders.values():
        Path(folder_path).mkdir(parents=True, exist_ok=True)
    return folders

# --- Variables Globales ---

STRUCTURE = create_fast_app_structure()
DB_PATH = os.path.join(STRUCTURE["db"], "fast_label_db.json")
THUMBNAILS_FOLDER = STRUCTURE["thumbnails"]

# --- Fonctions de base de données ---

def load_db():
    """Charge la base de données depuis le fichier JSON."""
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_db(data):
    """Sauvegarde les données dans le fichier JSON."""
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def is_rush_processed(video_path):
    """Vérifie si un rush a déjà été traité en se basant sur son chemin."""
    db = load_db()
    return video_path in db

# --- Fonctions d'analyse ---

def extract_middle_frame_and_duration(video_path):
    """
    Extrait la frame du milieu d'une vidéo, la sauvegarde comme thumbnail,
    et retourne le chemin de la thumbnail ainsi que la durée de la vidéo.
    """
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        duration_in_seconds = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration_in_seconds = float(stream.duration * stream.time_base)

        # Si la durée est nulle, essayer avec la durée du conteneur
        if duration_in_seconds == 0.0 and container.duration is not None:
            duration_in_seconds = float(container.duration / 1000000) # AV_TIME_BASE is 1_000_000

        # Se positionner au milieu de la vidéo
        # On cherche en utilisant un timestamp, qui est un entier.
        # On se base sur la durée du stream en sa propre base de temps.
        if stream.duration is not None:
            middle_timestamp = stream.duration // 2
            container.seek(middle_timestamp, stream=stream)
        
        frame = next(container.decode(video=0))
        
        # Créer la thumbnail
        img = frame.to_image()
        
        # Utiliser un nom de fichier encodé pour éviter les conflits
        encoded_path = base64.urlsafe_b64encode(video_path.encode()).decode()
        thumbnail_path = os.path.join(THUMBNAILS_FOLDER, f"{encoded_path}.jpg")
        
        img.save(thumbnail_path, "JPEG", quality=85)
        
        container.close()
        
        return thumbnail_path, duration_in_seconds
        
    except Exception as e:
        print(f"Erreur lors de l'extraction de la frame pour {video_path}: {e}")
        return None, None

def get_video_metadata(video_path):
    """Récupère les métadonnées d'une vidéo."""
    try:
        media_info = MediaInfo.parse(video_path)
        video_track = next((t for t in media_info.tracks if t.track_type == 'Video'), None)
        
        if video_track:
            return {
                "resolution": f"{video_track.width}x{video_track.height}",
                "frame_rate": float(video_track.frame_rate) if video_track.frame_rate else None,
                "format": video_track.format,
            }
    except Exception as e:
        print(f"Erreur lors de la récupération des métadonnées pour {video_path}: {e}")
    return {}

def labelize_frame(frame_path, api_key):
    """Appelle l'API de labellisation pour une image donnée."""
    try:
        with open(frame_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"base64_image": base64_image, "mode": "low"}
        
        response = requests.post(f"{config.API_URL}/labelize-image-fast", json=data, headers=headers)
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Erreur d'API lors de la labellisation de {frame_path}: {e}")
        return None
    except Exception as e:
        print(f"Erreur inattendue lors de la labellisation: {e}")
        return None

# --- Fonction Principale ---

def main_fast_labelize(video_path, api_key):
    """
    Fonction principale pour traiter un fichier vidéo rapidement.
    """
    config.API_STATUS = "Vérification du rush..."
    if not os.path.exists(video_path):
        print(f"Le fichier {video_path} n'existe pas.")
        config.API_STATUS = "Erreur: Fichier non trouvé"
        return {"error": "File not found"}

    if is_rush_processed(video_path):
        print(f"Le rush {video_path} a déjà été traité.")
        config.API_STATUS = "Rush déjà analysé"
        return {"status": "Already processed", "data": load_db()[video_path]}

    # --- Lancement de l'analyse ---
    
    # 1. Extraction de la frame et de la durée
    config.API_STATUS = "Extraction de la frame..."
    thumbnail_path, duration = extract_middle_frame_and_duration(video_path)
    if not thumbnail_path:
        config.API_STATUS = "Erreur d'extraction"
        return {"error": "Frame extraction failed"}

    # 2. Labellisation via API
    config.API_STATUS = "Labellisation en cours..."
    label_data = labelize_frame(thumbnail_path, api_key)
    if not label_data:
        config.API_STATUS = "Erreur de labellisation"
        return {"error": "Labeling API call failed"}
        
    # 3. Récupération des métadonnées
    config.API_STATUS = "Analyse des métadonnées..."
    metadata = get_video_metadata(video_path)
    
    # --- Compilation et sauvegarde ---
    
    result = {
        "file_path": video_path,
        "thumbnail_path": thumbnail_path,
        "duration": duration,
        "labels": label_data,
        "metadata": metadata,
        "status": "processed"
    }
    
    db = load_db()
    db[video_path] = result
    save_db(db)
    
    config.API_STATUS = "Analyse terminée !"
    print(f"Analyse de {video_path} terminée et sauvegardée.")
    
    return result

