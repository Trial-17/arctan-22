import os
import json
import librosa
import numpy as np
from typing import Dict
from pathlib import Path
import requests
from LIB import config
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


def load_music_analysis(analysis_path, audio_path):
    """
    Charge l'analyse musicale JSON, calcule l'intensité RMS à chaque beat,
    et retourne une liste enrichie avec le temps, type (beat/downbeat), segment, et intensité.

    Args:
        analysis_path (str): Chemin vers le fichier JSON contenant l'analyse musicale.
        audio_path (str): Chemin vers le fichier audio (.wav) correspondant.

    Returns:
        list: Liste de dictionnaires contenant time, type, segment, intensity.
    """
    if not os.path.exists(analysis_path):
        raise FileNotFoundError(f"Fichier non trouvé : {analysis_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Fichier audio non trouvé : {audio_path}")

    # Lecture du JSON
    with open(analysis_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            beats = data.get("beats", [])
            downbeats = set(data.get("downbeats", []))


            # Lecture de l'audio
            y, sr = librosa.load(audio_path, sr=None)
            frame_length = int(0.2 * sr)  # Fenêtre de 200 ms
            hop_length = int(0.05 * sr)   # Saut de 50 ms
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            rms_times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

            def get_rms_at_time(t):
                idx = np.argmin(np.abs(rms_times - t))
                return rms[idx]

            max_rms = np.max(rms)

            # Construction de la structure enrichie
            enriched_beats = []
            for beat_time in beats:
                beat_type = "downbeat" if beat_time in downbeats else "beat"
                if beat_type == "downbeat": 
                    intensity = get_rms_at_time(beat_time) / max_rms  # Normalisé entre 0 et 1
                    enriched_beats.append({
                        "time": beat_time,
                        # "type": beat_type,
                        "intensity": intensity
                    })

            return enriched_beats

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Erreur de lecture du JSON : {e}", doc=e.doc, pos=e.pos)

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

def define_cut_moments(music_data, user_prompt, rules, token, modele_choice):

    prompt =f"""

    {PROMPT_DEFINE_CUT_MOMENTS}
    ---
    
    ### Input : 
    
    - music_data: {music_data}
    - rules: {rules}
    - user_intent: {user_prompt}
    
    
    """.strip()
    modele = "gemini-2.5-pro-preview-05-06" if modele_choice == "PRO" else "gemini-2.5-flash-preview-05-20"

    try:
        response = requests.post(
            f"{config.API_URL}/global-gemini-call",
            json={"prompt": prompt,
                  "model": modele,
                  "schema_type" : "cuts"
                  },
            headers={"Authorization": f"Bearer {token}"}, 
            timeout=300 
        )
        
        if response.status_code != 200:
            print(response)
            print(f"Erreur lors de define cut: {response.status_code}")
            return {"cuts": []}  
            
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API de define cut {str(e)}")
        return {"cuts": []}  

def filter_music_data_by_cuts(music_data, cuts):
    """
    Filtre les beats de music_data en gardant uniquement ceux présents dans cuts.
    Si un temps de cut est absent, on le crée à partir du beat le plus proche.

    Args:
        music_data (list): Liste de beats enrichis avec 'time', 'type', 'segment', 'intensity'.
        cuts (list): Liste de temps en secondes (float) où un cut doit être effectué.

    Returns:
        list: Liste de beats correspondant uniquement aux cuts (originaux ou interpolés).
    """
    filtered = []
    time_to_beat = {round(item['time'], 2): item for item in music_data}

    for cut_time in cuts:
        cut_time_rounded = round(cut_time, 2)
        if cut_time_rounded in time_to_beat:
            filtered.append(time_to_beat[cut_time_rounded])
        else:
            # Trouve le beat le plus proche
            closest = min(music_data, key=lambda x: abs(x["time"] - cut_time))
            # Crée un nouveau beat basé sur le plus proche
            new_beat = closest.copy()
            new_beat["time"] = cut_time_rounded
            filtered.append(new_beat)

    return filtered

def define_plan_on_cut(montage_db, user_prompt, filtered_beats, chronologic_type, token, modele_choice):

    prompt = f"""
    
    ### Input : 
    
    - user_intent: {user_prompt}
    - montage_db: {montage_db}
    - filtered_beats : {filtered_beats}
    - chronologic_type: {chronologic_type}


    """.strip()
    
    modele = "gemini-2.5-pro-preview-05-06" if modele_choice == "PRO" else "gemini-2.5-flash-preview-05-20"
    try:
        response = requests.post(
            f"{config.API_URL}/global-gemini-call",
            json={"prompt": prompt,
                  "model": modele,
                  "schema_type" : "timeline"
                  },
            headers={"Authorization": f"Bearer {token}"}, 
            timeout=300 
        )
        
        if response.status_code != 200:
            print(f"Erreur lors de plan cut: {response.status_code}")
            return {"timeline": []}  
            
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API de plan cut: {str(e)}")
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

def auto_edit_music(user_prompt, rush_paths, analysis_path, audio_path, chronologic_type, token, modele_choice):
    """
    Fonction principale pour l'édition automatique de musique.
    """

    filtered_data, montage_db, effets_db, sound_db, file_mapping = load_filtered_db(GLOBAL_DB_PATH, rush_paths)
    music_data = load_music_analysis(analysis_path, audio_path)

    # cut_times = define_cut_moments(music_data, user_prompt,  CUT_INSTRUCTION, token, modele_choice)    
    # cut_times_filtered = filter_music_data_by_cuts(music_data, cut_times["cuts"])
    # config.API_STATUS = "Get editing tips for stage"
    # editing_instruction = get_editing_tips_for_stage(user_prompt, config.STEP_RUSH_SELECTION, token)
    config.API_STATUS = "Performing Editing"
    print("Performing Editing")
    timeline_V0  = define_plan_on_cut(montage_db, user_prompt, music_data, chronologic_type, token, modele_choice)   
    timeline_V1 = replace_rush_ids_with_paths(timeline_V0, file_mapping)
    
    return timeline_V1