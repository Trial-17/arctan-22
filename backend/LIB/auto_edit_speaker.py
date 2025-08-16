import os
import json
from typing import Dict
from pathlib import Path
import requests
from LIB import config
import re
import warnings
warnings.filterwarnings("ignore")

# ---- Environnement 

def create_app_structure() -> Dict[str, str]:
    """
    Crée la structure de dossiers de l'application dans Documents/Adobe/Premiere Pro/Premiere Copilot/
    Retourne un dictionnaire contenant les chemins de tous les dossiers créés.
    """
    # Chemin de base
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    
    # Structure des dossiers
    folders = {
        "thumbnails": str(base_path / "thumbnails"),
        "temp": str(base_path / "temp"),
        "rush_db": str(base_path / "rush_db"),

    }
    
    # Création des dossiers
    for folder_path in folders.values():
        Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    return folders

GLOBAL_DB_PATH = str( create_app_structure()['rush_db'] + "/rush_db.json")
GLOBAL_TEMP_FOLDER = str(create_app_structure()['temp'])



# ---- Fonctions temporaires 
def list_video_files(folder_path, extensions=None):
    """
    Liste tous les chemins de fichiers vidéo dans un dossier donné.

    Args:
        folder_path (str): Chemin vers le dossier à explorer.
        extensions (list, optional): Liste des extensions vidéo à rechercher (ex: ['.mp4', '.mov']).
                                     Si None, utilise une liste par défaut.

    Returns:
        list: Liste des chemins complets vers les fichiers vidéo trouvés.
    """
    if extensions is None:
        extensions = ['.mp4', '.mov', '.mxf', '.avi', '.mkv']

    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                video_files.append(os.path.join(root, file))

    return video_files

def load_filtered_db(global_db_path, rush_paths):
    """
    Charge la base de données JSON et filtre les éléments dont le chemin est présent dans rush_paths.
    Puis génère 3 versions : pour le montage, les effets et le sound design.
    Remplace aussi les chemins de fichiers dans la version montage par des identifiants "rush_i".

    Args:
        global_db_path (str): Chemin vers le fichier JSON de la base de données.
        rush_paths (list): Liste des chemins de fichiers vidéo à conserver.

    Returns:
        tuple: (filtered_data, montage_db, effets_db, sound_db, file_mapping)
    """
    if not os.path.exists(global_db_path):
        raise FileNotFoundError(f"Fichier non trouvé : {global_db_path}")
    
    with open(global_db_path, 'r', encoding='utf-8') as f:
        try:
            db_data = json.load(f)
            filtered_data = [item for item in db_data if item.get("file") in rush_paths]

            montage_db = []
            effets_db = []
            sound_db = []
            file_mapping = {}

            for i, item in enumerate(filtered_data):
                original_path = item.get("file", "")
                is_drone = "DJI" in os.path.basename(original_path)
                camera_type = "drone" if is_drone else "ground"
                rush_id = f"rush_{i}"
                file_mapping[rush_id] = original_path

                # 🎬 Montage version
                montage_entry = {
                    "file": rush_id,
                    "data": item["data"],
                    "Date": item.get("Date", ""),
                    "Camera": camera_type
                }
                montage_db.append(montage_entry)

                # 💥 Effets version
                effets_entry = {
                    k: v for k, v in item.items()
                    if k not in ["thumbnail", "Format", "Date"]
                }
                effets_entry["Camera"] = camera_type
                effets_db.append(effets_entry)

                # 🔊 Sound design version
                sound_entry = {
                    "file": original_path,
                    "data": item["data"],
                    "Camera": camera_type
                }
                sound_db.append(sound_entry)

            return filtered_data, montage_db, effets_db, sound_db, file_mapping

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Erreur de lecture du JSON : {e}", doc=e.doc, pos=e.pos)

def load_srt_file(srt_path):
    """
    Charge un fichier SRT et retourne une liste de blocs contenant start, end et texte.

    Args:
        srt_path (str): Chemin vers le fichier .srt

    Returns:
        list[dict]: Liste de segments avec 'start', 'end', 'text'
    """
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Découpage par blocs
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            timecode = lines[1]
            match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timecode)
            if match:
                start = match.group(1).replace(',', '.')
                end = match.group(2).replace(',', '.')
                text = " ".join(lines[2:]).strip()
                subtitles.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
    return subtitles

def define_plan_on_cut_SPEAKER(montage_db, user_prompt, audio_data, chronologic_type, token, modele_choice):

    prompt = f"""
     
    ### Input : 
    
    - montage_db: {montage_db}
    - audio_data : {audio_data}
    - user_intent: {user_prompt}
    - chronologic_type: {chronologic_type}


    """.strip()

    modele = "gemini-2.5-pro-preview-05-06" if modele_choice == "PRO" else "gemini-2.5-flash-preview-05-20"
    try:
        response = requests.post(
            f"{config.API_URL}/global-gemini-call",
            json={"prompt": prompt,
                  "model": modele,
                  "schema_type" : "timeline_speakers"
                  },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Erreur lors de l'amélioration du prompt: {response.status_code}")
            return {"timeline": []}  
            
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API d'amélioration de prompt: {str(e)}")
        return {"timeline": []}  

def replace_rush_ids_with_paths(timeline_result, file_mapping):
    """
    Remplace les identifiants 'rush_i' dans un résultat Gemini par les vrais chemins de fichier.

    Args:
        timeline_result (dict): Résultat Gemini avec une clé "timeline" contenant une liste de dicts.
            Chaque dict contient :
                - "time": float
                - "clip_file": str (ex: "rush_0")
        file_mapping (dict): Dictionnaire de correspondance { "rush_0": "/path/to/file.mov", ... }

    Returns:
        list: Nouvelle liste de timeline avec clip_file mis à jour avec les chemins absolus.
    """
    updated_timeline = []

    for entry in timeline_result.get("timeline", []):
        rush_id = entry["clip_file"]
        real_path = file_mapping.get(rush_id, rush_id)  # fallback au cas où non trouvé
        updated_entry = {
            **entry,
            "clip_file": real_path
        }
        updated_timeline.append(updated_entry)

    return updated_timeline

def auto_edit_speaker(user_prompt, rush_paths, srt_path, chronologic_type, token, modele_choice):
    
    filtered_data, montage_db, effets_db, sound_db, file_mapping = load_filtered_db(GLOBAL_DB_PATH, rush_paths)
    audio_data = load_srt_file(srt_path)

    # editing_instruction = get_editing_tips_for_stage(user_prompt, config.STEP_ADDING_BROLL, token)
    config.API_STATUS = "Defining Plan"
    timeline_V0  = define_plan_on_cut_SPEAKER(montage_db, user_prompt, audio_data, chronologic_type, token, modele_choice)
    timeline_V1 = replace_rush_ids_with_paths(timeline_V0, file_mapping)
    # print(timeline_V1)
    return timeline_V1
    
    