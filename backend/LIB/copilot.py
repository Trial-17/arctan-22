import json
from pathlib import Path
import json
import base64
import json
import copy
import os
import requests
from LIB import config

def create_script_directory():
    """
    Crée le dossier 'script' dans 'image_generation' ainsi que les fichiers JSON nécessaires.
    """
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "script"
    base_path.mkdir(parents=True, exist_ok=True)

    # Création des fichiers JSON vides s'ils n'existent pas
    for filename in ["history.json", "saved_script.json"]:
        file_path = base_path / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    return base_path

def add_to_history(id_prompt: str, prompt: str, code: str):
    """
    Ajoute une entrée à 'history.json' dans le dossier 'script'.
    """
    script_path = create_script_directory()
    history_file = script_path / "history.json"

    new_entry = {
        "id_prompt": id_prompt,
        "prompt": prompt,
        "code": code
    }

    with open(history_file, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(new_entry)
        f.seek(0)
        json.dump(data, f, indent=4)

def save_script_from_history(id_prompt: str, script_name: str):
    """
    Recherche un script dans l'historique par son ID et l'ajoute aux scripts sauvegardés.
    Retourne True si le script a été trouvé et sauvegardé, False sinon.
    """
    script_path = create_script_directory()
    history_file = script_path / "history.json"
    saved_file = script_path / "saved_script.json"
    
    # Recherche du script dans l'historique
    with open(history_file, "r", encoding="utf-8") as f:
        history_data = json.load(f)
        for entry in history_data:
            if entry["id_prompt"] == id_prompt:
                # Script trouvé, on l'ajoute aux scripts sauvegardés
                new_script = {
                    "id_prompt": id_prompt,
                    "name": script_name,
                    "code": entry["code"]
                }
                
                # Lecture et mise à jour du fichier saved_script.json
                with open(saved_file, "r+", encoding="utf-8") as sf:
                    saved_data = json.load(sf)
                    saved_data.append(new_script)
                    sf.seek(0)
                    json.dump(saved_data, sf, indent=4)
                
                return True
    
    # Script non trouvé
    return False

def delete_history_file():
    """
    Supprime le fichier 'history.json' s'il existe.
    """
    script_path = create_script_directory()
    history_file = script_path / "history.json"

    if history_file.exists():
        history_file.unlink()
        

# -------------- Fonctions Copilot --------------

# AudioSearch


def groc_transcription(audio_path, token):

    def encode_audio_base64(audio_path):
        with open(audio_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded

    audio_base64 = encode_audio_base64(audio_path)
    
    # ====== Remplacé dans l'API

    filename = os.path.basename(audio_path)
    url = f"{config.API_URL}/speaker_analysis"  
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"  
    }
    
    payload = {
        "audio_base64": audio_base64,
        "filename": filename
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  
    
    result = response.json()
    transcription = result["transcription"]


    # ======


    return transcription

def find_passages(segments_list, prompt ,token):

    # model = "gemini-2.5-flash-preview-04-17"  
    model = "gemini-2.5-pro-preview-05-06"  
    print(prompt)
    prompt = f"""

    ### User request :
    {prompt}
    
    ### Transcription to analyse :
    {segments_list}
        """.strip()
        
        
    try:
        response = requests.post(
            f"{config.API_URL}/global-gemini-call",
            json={"prompt": prompt,
                  "model": model,
                  "schema_type" : "audio_search"
                  },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Erreur lors de l'amélioration du prompt: {response.status_code}")
            return []
            
        return response.json()["passages"]
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API d'amélioration de prompt: {str(e)}")
        return []
    
def audio_search(audio_path, prompt, token):

    transcription = groc_transcription(audio_path, token)

    transcription_corrected = copy.deepcopy(transcription)

    segments_list = []

    for segment in transcription_corrected["segments"]:
        
        segments_list.append({  segment["start"] :segment["text"]})
    print(segments_list)     
    passages = find_passages(segments_list, prompt, token)

    return passages