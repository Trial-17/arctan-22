import os
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading
import signal
import uuid
import asyncio
from starlette.responses import StreamingResponse
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import requests
import json
from typing import Any, Dict
import logging



from LIB import config
from LIB.config import PENDING_JS_CALLS
from LIB.agent import run_agent_streaming
from LIB.podcast_V2 import main_podcast
from LIB.generation import main_generation, get_enhanced_prompt
from LIB.silence import analyser_et_visualiser_silence_amélioré
from LIB.subtitles import main_smart_srt, main_find_passages


# --- Initialisation
inactivity_timeout = 3600  
last_activity = time.time()  # Timestamp de la dernière activité
CHECK_INTERVAL_SECONDS = 10  # vérifie l'inactivité toutes les 10s
shutdown_lock = threading.Lock()  # Protection pour l'accès concurrent
shutdown_enabled = True  # Permet de désactiver le shutdown si nécessaire


def reset_shutdown_timer():
    """Réinitialise le timer de shutdown à 1h à partir de maintenant."""
    global last_activity
    with shutdown_lock:
        last_activity = time.time()
    print(f"🔄 Timer d'inactivité réinitialisé. Prochain arrêt dans {inactivity_timeout}s")

def shutdown_after_timeout():
    """Thread qui surveille l'inactivité et arrête l'API après le timeout."""
    global last_activity, shutdown_enabled
    # print(f"🚀 Thread de surveillance d'inactivité démarré (timeout: {inactivity_timeout}s)")
    
    while shutdown_enabled:
        with shutdown_lock:
            now = time.time()
            time_since_last_activity = now - last_activity
        
        if time_since_last_activity >= inactivity_timeout:
            print(f"⏳ Inactivité détectée ({int(time_since_last_activity)}s). Arrêt de l'API...")
            os.kill(os.getpid(), signal.SIGINT)
            return
        
        # Log périodique pour vérifier que le thread fonctionne (toutes les 5 minutes)
        if int(time_since_last_activity) % 300 == 0 and int(time_since_last_activity) > 0:
            remaining = int(inactivity_timeout - time_since_last_activity)
            print(f"⏱️  Inactivité: {int(time_since_last_activity)}s. Arrêt dans {remaining}s si pas d'activité.")
        
        time.sleep(CHECK_INTERVAL_SECONDS)



app = FastAPI(version="2.0.0",)

@app.on_event("startup")
def start_shutdown_monitor():
    """Démarre le thread de surveillance uniquement au lancement de l'app (worker)."""
    threading.Thread(target=shutdown_after_timeout, daemon=True).start()

# Middleware pour tracker automatiquement toutes les requêtes
@app.middleware("http")
async def track_activity_middleware(request: Request, call_next):
    """
    Middleware qui met à jour le timestamp à chaque requête HTTP.
    Cela permet d'éviter d'appeler manuellement reset_shutdown_timer() dans chaque endpoint.
    """
    reset_shutdown_timer()
    response = await call_next(request)
    return response






@app.get("/status")
def status():
    return {"status": config.API_STATUS}

@app.post("/stop-agent")
def stop_agent():
    """
    Endpoint pour arrêter l'agent en cours d'exécution.
    Met le flag STOP_REQUESTED à True.
    """
    config.STOP_REQUESTED = True
    config.API_STATUS = "End"
    return {"status": "stop_requested"}
 



# ========================================================
#                        COPILOT
# ========================================================


# --- Ajout pour la gestion de la mémoire ---
CONVERSATION_HISTORIES: Dict[str, List[Dict[str, Any]]] = {}
# -----------------------------------------

@app.get("/get-exposed-function")
async def get_exposed_function():
    """
    Polled by the JS client to check for functions to execute.
    Returns the first pending function found.
    """
    for call_id, data in PENDING_JS_CALLS.items():
        if data["status"] == "pending":
            data["status"] = "running"
            return {
                "args": data["args"],
                "call_id": call_id
            }
    return {}

class ResultRequest(BaseModel):
    call_id: str
    result: Any

@app.post("/receive-result")
async def receive_result(request: ResultRequest):
    """
    Called by the JS client to post the result of a function execution.
    """
    if request.call_id in PENDING_JS_CALLS:
        PENDING_JS_CALLS[request.call_id]["result"] = request.result
        PENDING_JS_CALLS[request.call_id]["status"] = "completed"
        return {"status": "ok"}
    return {"status": "error", "message": "call_id not found"}


# ------- Agent
class StreamChatRequest(BaseModel):
    prompt: str
    token: str
    conversation_id: Optional[str] = None
    model: Optional[str] = "fast"  # "fast" or "pro"

@app.post("/stream-chat")
async def stream_chat(request: StreamChatRequest):
    """
    Endpoint de chat qui retourne une réponse en streaming de l'agent
    et gère l'historique de la conversation.
    """
    # Réinitialiser le flag de stop au début de chaque requête
    config.STOP_REQUESTED = False
    print(f"--- Requête reçue sur /stream-chat avec le prompt: '{request.prompt}' et le modèle: '{request.model}' ---")
    config.API_STATUS = "Thinking..."
    conversation_id = request.conversation_id or str(uuid.uuid4())
    history = CONVERSATION_HISTORIES.get(conversation_id, [])

    async def event_generator():
        # D'abord, on envoie l'ID de la conversation au client s'il est nouveau
        if not request.conversation_id:
            yield json.dumps({"type": "conversation_start", "conversation_id": conversation_id}, ensure_ascii=False) + "\n"

        # L'agent va maintenant yield des dictionnaires qu'on transforme en JSON
        full_response_content = ""
        # Accumulation progressive du texte streamé (événements "thought") au cas où l'"answer" final serait vide
        streamed_text_accumulator = ""
        tool_calls = []

        async for item in run_agent_streaming(request.prompt, history, request.token, request.model):
            # On stream l'item au client
            yield json.dumps(item, ensure_ascii=False) + "\n"
            
            # On accumule le contenu pour la mémoire
            item_type = item.get("type")
            if item_type == "answer":
                # Réponse finale si fournie par l'agent (peut être vide selon les cas)
                content = item.get("content", "")
                if content:
                    full_response_content = content
            elif item_type == "thought":
                # Contenu streamé (morceaux de texte) : on les cumule en secours
                streamed_text_accumulator += item.get("content", "")
            elif item_type == "tool_start":
                 # Simplification: on ne stocke que le nom de l'outil appelé pour l'instant
                 tool_calls.append({"name": item.get("title"), "args": item.get("args")})


        # Une fois le stream terminé, on met à jour l'historique côté serveur
        try:
            # Création des messages à sauvegarder
            user_message = {"role": "user", "content": request.prompt}
            
            # Reconstruction du message de l'IA
            if not full_response_content.strip() and streamed_text_accumulator.strip():
                # Secours: si l'agent n'a pas émis d'"answer" non vide, on enregistre le texte streamé
                full_response_content = streamed_text_accumulator.strip()
            ai_message_content = {
                "content": full_response_content,
                "tool_calls": tool_calls # On ajoute les appels d'outils
            }
            ai_message = {"role": "assistant", "content": json.dumps(ai_message_content)}
            # print(f"Ai message: {ai_message}")
            # Mise à jour de l'historique
            current_history = CONVERSATION_HISTORIES.get(conversation_id, [])
            current_history.extend([user_message, ai_message])
            CONVERSATION_HISTORIES[conversation_id] = current_history
            # print(f"--- Historique de la conversation {conversation_id} mis à jour. Total d'échanges: {len(current_history)/2} ---")
            # print(CONVERSATION_HISTORIES)

        except Exception as e:
            print(f"Erreur lors de la mise à jour de l'historique: {e}")

    
    return StreamingResponse(event_generator(), media_type="text/plain")






# ========================================================
#                        PODCAST
# ========================================================




class PodcastRequest(BaseModel):
    paths: List[str]
    front_data: Dict[str, Any]
    token: str  # Ajout du token pour vérification d'accès

@app.post("/podcast")
def podcast(request: PodcastRequest):
    file_paths = request.paths
    front_data = request.front_data
    token = request.token

    try:
        result = main_podcast(file_paths, front_data, user_token=token)
        config.API_STATUS = "End"
        config.RESULTS = result
        return {"status": "started"}
    except Exception as e:
        config.API_STATUS = "Error"
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "technical",
                "message": f"An error occurred during podcast processing: {str(e)}\n\nPlease contact us at admin@premierecopilot.com, we will get back very quickly.",
                "technical_details": traceback.format_exc(),
                "error": str(e)
            }
        )


@app.get("/podcast-results")
def get_podcast_results():
    if config.API_STATUS != "End":
        return {"error": "Not ready"}

    print(config.RESULTS)
    return config.RESULTS 






# ========================================================
#                        GENERATIVE AI
# ========================================================





class GenerationRequest(BaseModel):
    token: str
    input: Dict[str, Any]

@app.post("/generation")
def generation(request: GenerationRequest):
    try:
        print(request.input)
        result = main_generation(request.input, request.token)
        
        if result is None:
            raise HTTPException(status_code=500, detail="La génération a échoué : aucun résultat retourné")
        
        # Si le résultat est déjà un dictionnaire (cas du voice cloning), le retourner tel quel
        if isinstance(result, dict):
            print(result)
            return result
        else:
            # Sinon, l'envelopper dans la structure standard
            print({"result": result})
            return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur lors de la génération: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {str(e)}")


class PromptEnhancementRequest(BaseModel):
    prompt: str
    token: str
    outputType: str = "image"  # Default to image if not specified
    
@app.post("/prompt-enhancement")
def prompt_enhancement(request: PromptEnhancementRequest):
    result = get_enhanced_prompt(request.prompt, request.token, request.outputType)
    return {"result": result}





# ========================================================
#                        JUMP CUT
# ========================================================


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

    try:
        time_segments, preview_image_path = analyser_et_visualiser_silence_amélioré(
            input_path=request.audio_path,
            db_threshold=request.silence_cutoff * (-1),
            min_silence_duration=request.remove_silences_over,
            min_segment_duration=request.keep_segments_over,
            padding=request.padding,
            offset=request.offset,
            preview=request.preview
        )
        
        return {
            "preview_image_path": preview_image_path,
            "time_segments": time_segments
        }
    except Exception as e:
        config.API_STATUS = "Error"
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "technical",
                "message": f"An error occurred during jump cut analysis: {str(e)}\n\nPlease contact us at admin@premierecopilot.com, we will get back very quickly.",
                "technical_details": traceback.format_exc(),
                "error": str(e)
            }
        )





# ========================================================
#                        SRT
# ========================================================



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
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "technical",
                "message": f"An error occurred during subtitle generation: {str(e)}\n\nPlease contact us at admin@premierecopilot.com, we will get back very quickly.",
                "technical_details": traceback.format_exc(),
                "error": str(e)
            }
        )

@app.get("/subtitles-results")
def get_subtitles_results():
    if config.API_STATUS == "End":
        return config.RESULTS
    elif config.API_STATUS == "Error":
        return config.RESULTS
    else:
        return {"status": "processing"}





# ========================================================
#                        AUDIORESEARCH
# ========================================================



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
    try:
        result = main_find_passages(request.audio_path, request.token, request.model, request.prompt)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "technical",
                "message": f"An error occurred during audio research: {str(e)}\n\nPlease contact us at admin@premierecopilot.com, we will get back very quickly.",
                "technical_details": traceback.format_exc(),
                "error": str(e)
            }
        )







# Filtre pour masquer les logs de polling
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Masquer les logs pour /get-exposed-function et /status
        message = record.getMessage()
        return message.find("/get-exposed-function") == -1 and message.find("/status") == -1 and message.find("receive-result") == -1

# Appliquer le filtre au logger uvicorn
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


if __name__ == "__main__":
    from api import app
    uvicorn.run(app, host="127.0.0.1", port=8000)
        

# if __name__ == "__main__":
#     uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)