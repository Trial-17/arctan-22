import os
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading
import signal


import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI

from pydantic import BaseModel
import uvicorn
import requests
import json
from typing import Any, Dict



from LIB import config
from LIB.music_analysis_V2 import analyze_music
from LIB.podcast_V2 import main_podcast
from LIB.image_generation import main_image_generation
from LIB.video_generation import generate_runway_video
from LIB.auto_fill import process_image_with_flux
from LIB.copilot import add_to_history, delete_history_file, save_script_from_history
from LIB.auto_edit_music import auto_edit_music
from LIB.speaker_analysis import analyze_speaker
from LIB.auto_edit_speaker import auto_edit_speaker

from LIB.silence import analyser_et_visualiser_silence_amélioré
from LIB.subtitles import main_smart_srt, main_find_passages
# from LIB.labelize_V3 import main_labelize
# from LIB.embedding_V3 import sync_vectorized_rush_db, load_vectorized_db, search_best_matches_vectorized
# --- Initialisation
inactivity_timeout = 3600  
last_activity_time = time.time()
# vectorized_db = load_vectorized_db()

def shutdown_after_timeout():
    global last_activity_time
    while True:
        time.sleep(3600)
        if time.time() - last_activity_time > inactivity_timeout:
            print("⏳ Inactivité détectée. Arrêt de l'API...")
            delete_history_file()
            os.kill(os.getpid(), signal.SIGINT)

threading.Thread(target=shutdown_after_timeout, daemon=True).start()

app = FastAPI(version="1.5.0",)





# ------- Outils
@app.middleware("http")
async def reset_inactivity_timer(request, call_next):
    global last_activity_time
    last_activity_time = time.time()
    response = await call_next(request)
    return response

@app.get("/hello")
def hello():
    return {"message": "Hello World"}

@app.get("/shutdown")
def shutdown():
    pid = os.getpid()
    threading.Thread(target=lambda: os.kill(pid, signal.SIGINT)).start()
    return {"message": "Shutting down"}


@app.get("/status")
def status():
    return {"status": config.API_STATUS}
 

# ------- Premiere Copilot
class PremiereCopilotRequest(BaseModel):
    id_prompt : str
    prompt: str
    top_k: int = 5
    token: str

@app.post("/premiere-copilot")
def premiere_copilot(request: PremiereCopilotRequest):
    id_prompt = request.id_prompt
    prompt = request.prompt
    token = request.token
    top_k = request.top_k
    
    try:
        # Appel à l'API externe
        response = requests.post(
            f"{config.API_URL}/premiere-copilot",
            json={"prompt": prompt, "top_k": top_k},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            return {"code": "", "answer": f"Erreur de l'API: {response.status_code}"}
        
        add_to_history(id_prompt, prompt,  response.json()["code"])  
        return response.json()
        
    except Exception as e:
        return {"code": "", "answer": f"Erreur: {str(e)}"}



class SaveScriptRequest(BaseModel):
    id_prompt: str
    script_name: str

@app.post("/save-script")
def save_script(request: SaveScriptRequest):
    result = save_script_from_history(request.id_prompt, request.script_name)
    if result:
        return {"status": "success", "message": "Script sauvegardé avec succès"}
    else:
        return {"status": "error", "message": "Script non trouvé dans l'historique"}

@app.get("/get-saved-scripts")
def get_saved_scripts():
    """
    Récupère la liste des scripts sauvegardés.
    """
    script_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "script"
    saved_file = script_path / "saved_script.json"
    
    if not saved_file.exists():
        return []
    
    with open(saved_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

class ExecuteScriptRequest(BaseModel):
    id_prompt: str

@app.post("/execute-script")
def execute_script(request: ExecuteScriptRequest):
    """
    Exécute un script en retrouvant son code depuis saved_script.json.
    """
    script_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "script"
    saved_file = script_path / "saved_script.json"
    
    if not saved_file.exists():
        return {"status": "error", "message": "Aucun script sauvegardé"}
    
    with open(saved_file, "r", encoding="utf-8") as f:
        scripts = json.load(f)
        for script in scripts:
            if script.get("id_prompt") == request.id_prompt:
                return {"status": "success", "code": script["code"]}
    
    return {"status": "error", "message": "Script non trouvé"}



# --- Ajout pour la communication avec les outils JS

import asyncio
from starlette.responses import StreamingResponse
from LIB.shared_state import JS_TOOL_CALLS

from LIB.fast_label import main_fast_labelize

from LIB.agent import run_agent_streaming

class JsResult(BaseModel):
    call_id: str
    result: Any

@app.post("/js-result")
def post_js_result(data: JsResult):
    """
    Endpoint pour que le client JS envoie le résultat d'une exécution de fonction.
    """
    print(data)
    call_info = JS_TOOL_CALLS.get(data.call_id)
    if call_info and 'event' in call_info:
        call_info['result'] = data.result
        call_info['event'].set()
        return {"status": "ok"}
    return {"status": "error", "message": "call_id not found"}


# ------- Agent
class StreamChatRequest(BaseModel):
    prompt: str
    google_api_key: str

@app.post("/stream-chat")
async def stream_chat(request: StreamChatRequest):
    """
    Endpoint de chat qui retourne une réponse en streaming de l'agent.
    """
    print(f"--- Requête reçue sur /stream-chat avec le prompt: '{request.prompt}' ---")
    async def event_generator():
        # L'agent va maintenant yield des dictionnaires qu'on transforme en JSON
        async for item in run_agent_streaming(request.prompt, request.google_api_key):
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


# ------- Podcast
class PodcastRequest(BaseModel):
    paths: List[str]
    front_data: Dict[str, Any]
    token: str  # Ajout du token pour vérification d'accès

@app.post("/podcast")
def podcast(request: PodcastRequest):
    file_paths = request.paths
    front_data = request.front_data
    token = request.token  # Récupération du token

    result = main_podcast(file_paths, front_data, user_token=token)
    config.API_STATUS = "End"
    config.RESULTS = result
    return {"status": "started"}

@app.get("/podcast-results")
def get_podcast_results():
    if config.API_STATUS != "End":
        return {"error": "Not ready"}
    return config.RESULTS 
  
  
# ------- Labelisation 
# class LabelizeRequest(BaseModel):
#     paths: List[str]
#     token: str

# @app.post("/labelize")
# def labelize(request: LabelizeRequest):
#     config.API_STATUS = "Labelization"
#     global vectorized_db 
    
#     result = main_labelize(request.paths, request.token)
#     sync_vectorized_rush_db()
#     vectorized_db = load_vectorized_db()
#     config.API_STATUS = "End"
    
#     return {str(result)}

class FastLabelizeRequest(BaseModel):
    video_path: str
    token: str

@app.post("/fast-labelize")
def fast_labelize(request: FastLabelizeRequest):
    """
    Lance une analyse rapide pour un seul fichier vidéo.
    """
    config.API_STATUS = "Fast Labelization"
    result = main_fast_labelize(request.video_path, request.token)
    config.API_STATUS = "End"
    return result


# ------- Deep Research
# class SearchQuery(BaseModel):
#     query: str
#     top_k: int = 10

# @app.post("/search")
# def search(query: SearchQuery):

#     # results = search_rushes(query.query,  top_k=query.top_k)
#     results = search_best_matches_vectorized(vectorized_db, query.query , top_k=query.top_k, threshold=config.EMBEDDING_THRESHOLD)
#     config.API_STATUS = "Ready"
#     return {"results": results}


# ------- Image Generation
class ImageGenerationRequest(BaseModel):
    prompt: str
    image_path: Optional[str] = ""
    aspect: Optional[str] = "21:9"
    token: str
     
@app.post("/generate-image")
def generate_image(request: ImageGenerationRequest): 
    config.API_STATUS = "Generating Image"
    
    try:
        result = { "image_path": main_image_generation(request.token, request.image_path,  request.prompt, aspect=request.aspect) }
        config.API_STATUS = "End"
        return result
    except Exception as e:
        config.API_STATUS = "Error"
        print(f"❌ Erreur lors de la génération d'image: {str(e)}")
        return { "error": str(e) }


# ------- Video Generation 
class VideoGenerationRequest(BaseModel):
    token: str
    image_path: str 
    prompt: str
    ratio: str
    type: str
    
@app.post("/generate-video")
def generate_video(request: VideoGenerationRequest):
    
    try:
        print(f"🔄 Génération de vidéo avec les paramètres: {request.image_path}, {request.prompt}, {request.ratio}, {request.type}")
        output_path = generate_runway_video(request.token, request.image_path, request.prompt, request.ratio, request.type)

        return output_path
    except Exception as e:
        print(f"❌ Erreur lors de la génération vidéo: {str(e)}")
        return {"error": str(e)}


# ------- Auto Edit
class MusicAnalysisRequest(BaseModel):
    audio_path : str
    modele_choice : str

@app.post("/music-analysis")
def music_analysis(request: MusicAnalysisRequest):
    config.API_STATUS = "Music Analysis (realtime)"
    
    try:
        downbeats, beats, stem_folder, json_path= analyze_music(request.audio_path, request.modele_choice)

        config.RESULTS = {
            "downbeats": downbeats,
            "beats": beats if request.modele_choice == "PRO" else [],
            "demixed_path": stem_folder,
            "path_analysis": json_path
        }

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse audio : {e}")
        
        config.RESULTS = {
            "downbeats": [],
            "beats": [],
            "demixed_path": None,
            "path_analysis": None
        }

    finally:
        config.API_STATUS = "End"

    return {"status": "started"}

class SpeakerAnalysisRequest(BaseModel):
    prompt: str
    audio_path: str
    token: str
    modele_choice : str

@app.post("/speaker-analysis")
def speaker_analysis(request: SpeakerAnalysisRequest):
    config.API_STATUS = "Speaker Analysis (realtime)"
    result = analyze_speaker(request.prompt, request.audio_path, request.token, request.modele_choice)
    config.API_STATUS = "End"
    config.RESULTS = result

@app.get("/audio-results")
def get_audio_results():
    if config.API_STATUS != "End":
        return {"error": "Not ready"}
    return config.RESULTS

class AutoEditMusicRequest(BaseModel):
    prompt: str
    rush_paths: List[str]
    analysis_path: str
    audio_path: str
    chronologic_type: str
    token: str
    modele_choice: str

@app.post("/auto-edit-music")
def handle_auto_edit_music(request: AutoEditMusicRequest):
    config.API_STATUS = "Auto Edit Music"

    result = auto_edit_music(request.prompt, request.rush_paths, request.analysis_path, request.audio_path, request.chronologic_type, request.token, request.modele_choice)
    config.API_STATUS = "End"
    config.RESULTS = result

@app.get("/auto-edit-results")
def get_auto_edit_results():
    if config.API_STATUS != "End":
        return {"error": "Not ready"}
    return config.RESULTS 

class AutoEditSpeakerRequest(BaseModel):
    rush_paths: List[str]
    srt_path: str
    prompt: str
    chronologic_type: str
    token: str
    modele_choice : str

@app.post("/auto-edit-speaker")
def handle_auto_edit_speaker(request: AutoEditSpeakerRequest):
    config.API_STATUS = "Auto Edit Speaker"

    result = auto_edit_speaker(request.prompt, request.rush_paths, request.srt_path, request.chronologic_type, request.token, request.modele_choice)
    config.API_STATUS = "End"
    config.RESULTS = result

# ------- Jump Cut
class JumpCutPreviewRequest(BaseModel):
    silence_cutoff: float
    remove_silences_over: float
    keep_segments_over: float
    padding: float
    offset: float
    audio_path: str
    token: str
    preview: bool = True
    
@app.post("/jump-cut-preview")
def jump_cut_preview(request: JumpCutPreviewRequest):
    """
    Génère un aperçu pour la fonctionnalité Jump Cut en utilisant la fonction d'analyse.
    """
    print(f"Received jump cut preview request with params: {request}")

    if request.token is not None:
        try:
            headers = {"Authorization": f"Bearer {request.token}"}
            access_url = f"{config.API_URL}/user/podcast-access"
            resp = requests.get(access_url, headers=headers)
            resp.raise_for_status()
            access_data = resp.json()
            if not access_data.get("podcast_access", False):
                raise PermissionError("L'utilisateur n'a pas accès au podcast (abonnement insuffisant).")

        except Exception as e:
            raise PermissionError(f"Erreur lors de la vérification d'accès podcast: {e}")

    
    try:
        time_segments, preview_image_path = analyser_et_visualiser_silence_amélioré(
            input_path=request.audio_path,
            db_threshold=request.silence_cutoff * (-1),
            min_silence_duration=request.remove_silences_over,
            min_segment_duration=request.keep_segments_over, # Note: This might not be the right parameter name from the front-end
            padding=request.padding,
            offset=request.offset,
            preview=request.preview
        )
        
        return {
            "preview_image_path": preview_image_path,
            "time_segments": time_segments
        }
    except Exception as e:
        print(f"Error during jump cut preview generation: {e}")
        # Return a proper error response
        return {"error": f"Failed to generate preview: {str(e)}"}, 500


# ------- Flux Fill
class FluxFillRequest(BaseModel):
    token: str
    image_path: str
    prompt: str
    
@app.post("/flux-fill")
def flux_fill(request: FluxFillRequest):
    config.API_STATUS = "Flux Fill"
    
    result = process_image_with_flux(request.token, request.image_path, request.prompt)
    config.API_STATUS = "End"
    return result



# ------- Subtitles
class SubtitlesRequest(BaseModel):
    audio_path: str
    model: str
    user_prompt: str
    max_caracteres: int = 60
    seuil: float = 0.15
    lignes_max_par_srt: int = 2
    ponctuation_force_solo: bool = True
    token: str
    
@app.post("/subtitles")
def subtitles(request: SubtitlesRequest):
    config.API_STATUS = "Subtitles"
    print(f"Received subtitles request with params: {request}")
    try:
        result = main_smart_srt(
            request.audio_path, request.token, request.model, request.user_prompt, 
            request.max_caracteres, request.seuil, request.lignes_max_par_srt, 
            request.ponctuation_force_solo
        )
        config.API_STATUS = "End"
        return {"srt_path": result}
    except Exception as e:
        config.API_STATUS = "Error API"
        return {"error": str(e)}

@app.get("/subtitles-results")
def get_subtitles_results():
    if config.API_STATUS == "End":
        return config.RESULTS
    elif config.API_STATUS == "Error":
        return config.RESULTS
    else:
        return {"status": "processing"}

class AudioResearchRequest(BaseModel):
    audio_path: str
    prompt : str
    token : str
    model : str
        
@app.post("/audio-research")
def audio_research(request: AudioResearchRequest):
    """
    Recherche audio dans un dossier donné.
    """

    return main_find_passages(request.audio_path,  request.token, request.model, request.prompt,)







if __name__ == "__main__":
    from api import app
    uvicorn.run(app, host="127.0.0.1", port=8000)
        

# if __name__ == "__main__":
    # uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)