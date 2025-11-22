import json
from typing import TypedDict, Annotated, List, Dict, Any
import operator
import time
import uuid
import requests
import base64
import asyncio
import os
import re
import shutil
from enum import Enum
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage, HumanMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import av

from LIB import config
from LIB.config import PENDING_JS_CALLS, EDIT_PROJECT_STRUCTURE_TOOL_LIST, MODEL_REACT_PROJECT_STRUCTURE, MODEL_REACT_PROJECT_STRUCTURE_TOOL
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from LIB.music_analysis_V2 import analyze_music
from LIB.subtitles import main_transcription_for_agent
from LIB.custom_llm import PremiereGPT_LLM
from LIB.gemini_logger import log_gemini_call


def format_agent_history(agent_history):
    """
    Formate l'historique de l'agent pour n'extraire que les contenus utiles.
    Filtre SystemMessage et ne garde que User/Assistant.
    
    Args:
        agent_history: Liste de messages LangChain
        
    Returns:
        str: Historique formaté proprement
    """
    if not agent_history:
        return "No previous conversation"
    
    formatted_lines = []
    for msg in agent_history:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"**User:** {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"**Assistant:** {msg.content}")
        # On ignore les SystemMessage
    
    return "\n\n".join(formatted_lines) if formatted_lines else "No previous conversation"


MODEL_CONTEXT_1 = "gemini-2.5-flash-lite" # Get Project Context


def gemini_call(prompt, system_instruction, structured_output = None, tool_list = None, temperature= 0.1,  model = "gemini-2.5-flash-lite"): 
    """
    Appelle l'API Gemini via l'API relai.
    Conserve la même interface et le même comportement que l'ancienne fonction.
    """
    
    
    try:
        # S'assurer que system_instruction est une chaîne de caractères
        if isinstance(system_instruction, list):
            system_instruction = " ".join(system_instruction)
        elif system_instruction is None:
            system_instruction = ""
        
        # Préparer la requête pour l'API relai
        payload = {
            "prompt": str(prompt),
            "system_instruction": str(system_instruction),
            "temperature": temperature,
            "model": model
        }
        
        # Ajouter les paramètres optionnels
        if structured_output:
            payload["structured_output"] = structured_output
        if tool_list:
            payload["tool_list"] = tool_list
        
        # Appeler l'API relai
        response = requests.post(
            f"{config.API_URL}/gemini-call",
            json=payload,
            headers={"Authorization": f"Bearer {config.AGENT_TOKEN}"}
        )
        
        # Vérifier le statut de la réponse
        if response.status_code != 200:
            raise Exception(f"Erreur API (status {response.status_code}): {response.text}")
        
        # Extraire le résultat
        result = response.json().get("result")
        
        log_gemini_call("agent.py::gemini_call", payload, result)  # LOG
        
        # Retourner le résultat dans le même format que l'ancienne fonction
        return result
        
    except requests.exceptions.RequestException as e:
        log_gemini_call("agent.py::gemini_call", payload if 'payload' in locals() else {}, None, str(e))  # LOG
        print(f"❌ Erreur lors de l'appel à l'API relai: {str(e)}")
        raise Exception(f"Erreur lors de l'appel à l'API relai: {str(e)}")







# ------------- PROJECT MANAGEMENT -------------

async def main_fast_labelize(list_clip):

    try : 
            
        # 1. Création / importation de la base
        path_clip_db = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "index.json"

        # Suppression de la base de données pour les tests
        # if path_clip_db.exists():
        #     path_clip_db.unlink()
        #     print(f"Base de données supprimée pour les tests : {path_clip_db}")

            
        if not path_clip_db.exists():
            path_clip_db.parent.mkdir(parents=True, exist_ok=True)
            with open(path_clip_db, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            print(f"Fichier index.json créé : {path_clip_db}")

        # Chargement de la base de données
        with open(path_clip_db, 'r', encoding='utf-8') as f:
            clip_db = json.load(f)
            
            

        # 2. Vérification des clip à labeliser
        existing_media_paths = set()

        for item in clip_db:
            if isinstance(item, dict) and 'mediaPath' in item:
                existing_media_paths.add(item['mediaPath'])


        clips_to_process = []
        for clip in list_clip:
            if clip not in existing_media_paths:
                clips_to_process.append(clip)


        list_clip = clips_to_process
        print(f"{len(clips_to_process)} clips à traiter sur {len(list_clip)} au total")




        # 3. Labelisation des clips
        cpt = 1
        for clip in list_clip:
            config.API_STATUS = "Indexing clip : " + str(cpt) + "/" + str(len(list_clip))
            print(config.API_STATUS)
            await asyncio.sleep(0.01)  # Laisse le temps au serveur de respirer
            cpt += 1
            # 3.A SI le clip est une video, extraire une frame 
            if clip.endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv')):
                
                thumb_labelize = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "thumb_labelize.jpg"

                container = av.open(clip)
                video_stream = container.streams.video[0]
                duration = float(video_stream.duration * video_stream.time_base)
                target_time = duration / 2
                container.seek(int(target_time / video_stream.time_base), stream=video_stream)

                for frame in container.decode(video_stream):
                    frame_resized = frame.reformat(height=720)
                    frame.to_image().save(thumb_labelize)
                    break

                container.close()
                image_path = thumb_labelize
                
            else: 
                image_path = clip


        

            
            # Lire l'image directement en binaire et encoder en base64
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            # Encoder l'image en base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Déterminer le mime_type en fonction de l'extension
            if image_path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/png' 
                
            # 3.B Labelisation de l'image   ----- APPEL API    
            
            # Préparer la requête pour l'API
            api_url = config.API_URL + "/labelize-image"
            headers = {
                "Authorization": f"Bearer {config.AGENT_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "base64_image": base64_image,
                "mime_type": mime_type
            }
            
            # Appeler l'API
            response = requests.post(api_url, json=payload, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Erreur API labelization: {response.status_code} - {response.text}")
            
            # 3.C Sauvegarder les labels
            labels = response.json()
            clip_db.append({
                "mediaPath": clip,
                "camera_angle": labels["camera_angle"],
                "colors": labels["colors"],
                "description": labels["description"],
                "people": labels["people"],
            })


        # 4. Sauvegarder la base

        with open(path_clip_db, 'w', encoding='utf-8') as f:
            json.dump(clip_db, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Erreur main_fast_labelize: {str(e)}")
        return "Error: " + str(e)
    return "done"

class GetProjectStructure(BaseModel):
    include_metadata :bool = Field(default=False, description="Use it only if you need to know the description of the video and images to perform the task. Slower")
 
@tool("get_project_structure", args_schema=GetProjectStructure)
async def get_project_structure(include_metadata: bool = False, skip_labelize: bool = False):
    """
    Returns the JSON structure of the Premiere Pro project
    Usefull ot get the availables clips, musics, audio, folders
    """
    try : 
        call_id = str(uuid.uuid4())

        config.API_STATUS = "Getting project structure..."

        
        # 1. Création / importation de la base
        if include_metadata:
            path_clip_db = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "index.json"
            
        
        # 1. Récupération de la structure du projet
        PENDING_JS_CALLS[call_id] = {
            "args": {"script": "$._MYFUNCTIONS.exportProjectStructureToJSON();"},
            "result": None,
            "status": "pending"
        }
        
        timeout = 30
        start_time = time.time()
        result = None
        while time.time() - start_time < timeout:
            if call_id in PENDING_JS_CALLS and PENDING_JS_CALLS[call_id]["status"] == "completed":
                result = PENDING_JS_CALLS[call_id]["result"]
                del PENDING_JS_CALLS[call_id]
                break
            await asyncio.sleep(0.5)
            
        if call_id in PENDING_JS_CALLS:
            del PENDING_JS_CALLS[call_id]

        if result is None: 
            return "Error: Timed out waiting for JavaScript function to execute."
        
        
        # 2. Enregistrement du projet (on peut écraser sans soucis)
        base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
        Path(base_path).mkdir(parents=True, exist_ok=True)

        project_data = json.loads(result)
        project_id = project_data.get("projectID", "")
        file_path = base_path / f"{project_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)



        
        
        # 3. On récupère tous les paths des fichiers images et vidéos et audios
        def extract_media_paths(item, paths_list):
            """Parcourt récursivement un élément et extrait tous les paths des médias vidéo et image"""
            if isinstance(item, dict):
                # Vérifier si l'élément actuel est une vidéo ou une image
                item_type = item.get('type', '')
                if item_type in ['Video', 'Image', 'Audio'] and 'mediaPath' in item:
                    media_path = item['mediaPath']
                    if media_path and media_path != "N/A":
                        paths_list.append({
                            'name': item.get('name', ''),
                            'type': item_type,
                            'path': media_path,
                            'nodeId': item.get('nodeId', '')
                        })
                
                # Parcourir récursivement les enfants s'ils existent
                if 'children' in item and isinstance(item['children'], list):
                    for child in item['children']:
                        extract_media_paths(child, paths_list)
            elif isinstance(item, list):
                # Si c'est une liste, parcourir chaque élément
                for child in item:
                    extract_media_paths(child, paths_list)
        
        media_paths = []
        extract_media_paths(project_data, media_paths)
        # print(f"Trouvé {len(media_paths)} fichiers média (Video/Image)")

        
        
        
        # 4. On cherche les metadatas de ces fichiers médias, sinon on appelle le tool labelizer
        path_clip_db = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "index.json"
        
        # Créer le fichier index.json s'il n'existe pas
        if not path_clip_db.exists():
            path_clip_db.parent.mkdir(parents=True, exist_ok=True)
            with open(path_clip_db, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)

        
        # Charger la base de données des rushs
        with open(path_clip_db, 'r', encoding='utf-8') as f:
            clip_db = json.load(f)

        existing_paths_in_db = {item['mediaPath'] for item in clip_db}

        missing_media = []
        for media in media_paths:
            media_path = media['path']

            if media_path not in existing_paths_in_db:
                missing_media.append(media_path)

        # if missing_media and include_metadata and not skip_labelize:
        #     print(f"\nRésumé : {len(missing_media)} fichiers nécessitent une analyse des métadonnées")
        #     await main_fast_labelize(missing_media)
            
        
        
        
        # 5. On ajoute les metadatas si demandé au result final
        with open(path_clip_db, 'r', encoding='utf-8') as f:
            clip_db = json.load(f)
        
        metadata_map = {item['mediaPath']: item for item in clip_db}

        def add_metadata_to_project_structure(item):
            """Parcourt récursivement la structure du projet pour y ajouter les métadonnées."""
            if isinstance(item, dict):
                item_type = item.get('type', '')
                
                if include_metadata and item_type in ['Video', 'Image'] and item['mediaPath'] in metadata_map:
                    meta = metadata_map[item['mediaPath']]
                    item['metadata'] = {
                        "description": meta.get("description"),
                        "camera_angle": meta.get("camera_angle"),
                        "colors": meta.get("colors"),
                        "people": meta.get("people")
                    }
                elif item_type == 'Audio' and item['mediaPath'] in metadata_map:
                    meta = metadata_map[item['mediaPath']]
                    item['metadata'] = {
                        "downbeats": meta.get("downbeats", None)
                    }
                
                if 'children' in item and isinstance(item['children'], list):
                    for child in item['children']:
                        add_metadata_to_project_structure(child)
            elif isinstance(item, list):
                for sub_item in item:
                    add_metadata_to_project_structure(sub_item)

        add_metadata_to_project_structure(project_data)
        final_result = json.dumps(project_data, indent=2, ensure_ascii=False)


        
        config.API_STATUS = "Thinking..."
        # print(json.dumps(project_data, indent=2))
        return final_result
    except Exception as e:
        print(f"Erreur get_project_structure: {str(e)}")
        return "Error: " + str(e)

async def get_project_context(prompt: str, include_metadata: bool = False):
    
    
    config.API_STATUS = "Getting project context..."

    # 1. Récupération de la structure du projet
    project_V0 = await get_project_structure.ainvoke({"include_metadata": include_metadata})
    

    # 2. Déterminer le contexte a envoyer
    full_prompt = f"""    
    ### JSON project :
    {project_V0}
    
    ### User prompt : 
    {prompt}
        """.strip()

    system_instruction="""
    You are a professional video editor. You receive a JSON representation of the project architecture.
    Your task is to select the right context that will be required to perform the user prompt.
    It will be then send to an agent that will defin alist of tasks based on this context.

    ### TASKS:
    - Analyse the user prompt
    - Select the right context that will be required to perform the user prompt and to understand the user intent
    - Return the full list of the ids of the context or 'all' if all the context is needed

    ### RULES:
    - if you want to include a folder, include it id, not its childrens
    - if you want to include a sequence, and audio, srt, video, image, include it id
    - if no context is needed, return an empty list
    - if all the context is needed, return a unique item id named "all"
        """, 

    structured_output = {
        "type": "array",
        "items": {
            "type": "string",
            "description": "The id of the bin, sequence, item to add to the context, or 'all' if all the context is needed"
        }
    }
            
    context = gemini_call(full_prompt, system_instruction, structured_output, None, 0.3, MODEL_CONTEXT_1)
        

    # 3. Retraitement du contexte : élimination des introuvés
    final_context = []
    processed_ids = set()  # Pour éviter les doublons
        
    if context:
        # Cas spécial : si "all" est présent
        if "all" in context:
            project_data = json.loads(project_V0)
            final_context = get_all_descendants(project_data, project_data.get('name', 'Project'))
        else:
            project_data = json.loads(project_V0)
            
            for item_id in context:
                # Vérifier si déjà traité (éviter doublons)
                if item_id in processed_ids:
                    continue
                
                # Chercher l'item dans le projet
                found_item = find_item_by_id(project_data, item_id, project_data.get('name', 'Project'))
                
                if found_item:
                    processed_ids.add(item_id)
                    
                    # Si c'est un Bin (dossier), inclure toute la descendance
                    if found_item.get("type") == "Bin":
                        descendants = get_all_descendants(found_item, found_item["path"])
                        final_context.extend(descendants)
                        
                        # Marquer tous les descendants comme traités pour éviter doublons
                        def mark_descendants_processed(node):
                            if node.get("nodeId"):
                                processed_ids.add(node.get("nodeId"))
                            if "children" in node:
                                for child in node["children"]:
                                    mark_descendants_processed(child)
                        
                        for desc in descendants:
                            mark_descendants_processed(desc)
                    else:
                        # Pour les autres types (Sequence, Audio, Video, etc.)
                        final_context.append(found_item)
                else:
                    # ID non trouvé, on l'ignore
                    print(f"Warning: ID {item_id} not found in project")
        
        # print("Final context:", json.dumps(final_context, indent=2))
    
    return final_context
  
def find_item_by_id(node, target_id, current_path=""):
    """Trouve un item par son ID et retourne l'item avec son path"""
    # Vérifier si le noeud actuel a l'ID recherché
    if node.get("nodeId") == target_id:
        item_copy = node.copy()
        item_copy["path"] = current_path
        return item_copy
    
    # Chercher dans les enfants si c'est un Bin ou Project
    if "children" in node:
        for child in node["children"]:
            child_path = f"{current_path}/{child.get('name', '')}" if current_path else child.get('name', '')
            result = find_item_by_id(child, target_id, child_path)
            if result:
                return result
    
        return None

def get_all_descendants(node, current_path=""):
    """Récupère tous les descendants d'un noeud avec leurs paths"""
    items = []
    node_copy = node.copy()
    
    # Ajouter le path au noeud actuel
    node_copy["path"] = current_path
    
    # Si le noeud a des enfants, les traiter
    if "children" in node:
        children_items = []
        for child in node["children"]:
            child_path = f"{current_path}/{child.get('name', '')}" if current_path else child.get('name', '')
            descendants = get_all_descendants(child, child_path)
            children_items.extend(descendants)
        
        # Remplacer les enfants par leur version avec paths
        node_copy["children"] = children_items
    return [node_copy]        

def compare_project_structures(initial_structure: str, final_structure: str) -> Dict:
    """
    Compare deux structures de projet et retourne les différences.
    Détecte: création de bins, création de séquences, suppression, modifications, renommages, déplacements
    """
    differences = {
        "bins_created": [],
        "sequences_created": [],
        "items_deleted": [],
        "sequences_updated": [],
        "items_renamed": [],
        "items_moved": []
    }
    
    try:
        initial_data = json.loads(initial_structure) if isinstance(initial_structure, str) else initial_structure
        final_data = json.loads(final_structure) if isinstance(final_structure, str) else final_structure
        
        # Créer des dictionnaires indexés par nodeId pour faciliter la comparaison
        def build_index(structure, index=None, parent_path=""):
            if index is None:
                index = {}
            
            if isinstance(structure, dict):
                if "nodeId" in structure:
                    node_id = structure["nodeId"]
                    index[node_id] = {
                        "name": structure.get("name", ""),
                        "type": structure.get("type", ""),
                        "path": parent_path,
                        "videoFrameWidth": structure.get("videoFrameWidth"),
                        "videoFrameHeight": structure.get("videoFrameHeight"),
                        "videoDisplayFormat": structure.get("videoDisplayFormat"),
                        "data": structure
                    }
                
                # Parcourir les enfants
                if "children" in structure:
                    current_path = f"{parent_path}/{structure.get('name', '')}" if parent_path else structure.get('name', '')
                    for child in structure["children"]:
                        build_index(child, index, current_path)
            
            elif isinstance(structure, list):
                for item in structure:
                    build_index(item, index, parent_path)
            
            return index
        
        initial_index = build_index(initial_data)
        final_index = build_index(final_data)
        
        # Détecter les éléments créés
        for node_id, node_info in final_index.items():
            if node_id not in initial_index:
                if node_info["type"] == "Bin":
                    differences["bins_created"].append({
                        "nodeId": node_id,
                        "name": node_info["name"],
                        "path": f"{node_info['path']}/{node_info['name']}" if node_info['path'] else node_info['name']
                    })
                elif node_info["type"] == "Sequence":
                    differences["sequences_created"].append({
                        "nodeId": node_id,
                        "name": node_info["name"],
                        "width": node_info["videoFrameWidth"],
                        "height": node_info["videoFrameHeight"],
                        "fps": node_info["videoDisplayFormat"]
                    })
        
        # Détecter les éléments supprimés
        for node_id, node_info in initial_index.items():
            if node_id not in final_index:
                differences["items_deleted"].append({
                    "nodeId": node_id,
                    "name": node_info["name"],
                    "type": node_info["type"]
                })
        
        # Détecter les modifications, renommages et déplacements
        for node_id in initial_index:
            if node_id in final_index:
                initial_item = initial_index[node_id]
                final_item = final_index[node_id]
                
                # Vérifier les renommages
                if initial_item["name"] != final_item["name"]:
                    differences["items_renamed"].append({
                        "nodeId": node_id,
                        "old_name": initial_item["name"],
                        "new_name": final_item["name"]
                    })
                
                # Vérifier les déplacements
                if initial_item["path"] != final_item["path"]:
                    differences["items_moved"].append({
                        "nodeId": node_id,
                        "name": final_item["name"],
                        "old_path": initial_item["path"] if initial_item["path"] else "/",
                        "new_path": final_item["path"] if final_item["path"] else "/"
                    })
                
                # Vérifier les modifications de séquences
                if initial_item["type"] == "Sequence":
                    changes = []
                    if initial_item["videoFrameWidth"] != final_item["videoFrameWidth"]:
                        changes.append(f"Largeur: {initial_item['videoFrameWidth']} → {final_item['videoFrameWidth']}")
                    if initial_item["videoFrameHeight"] != final_item["videoFrameHeight"]:
                        changes.append(f"Hauteur: {initial_item['videoFrameHeight']} → {final_item['videoFrameHeight']}")
                    if initial_item["videoDisplayFormat"] != final_item["videoDisplayFormat"]:
                        changes.append(f"FPS: {initial_item['videoDisplayFormat']} → {final_item['videoDisplayFormat']}")
                    
                    if changes:
                        differences["sequences_updated"].append({
                            "nodeId": node_id,
                            "name": final_item["name"],
                            "changes": changes
                        })
        
    except Exception as e:
        print(f"Erreur lors de la comparaison: {str(e)}")
        
    
    observation_parts = []
    
    if differences["bins_created"]:
        observation_parts.append(f"Created {len(differences['bins_created'])} bin(s)")
    if differences["sequences_created"]:
        observation_parts.append(f"Created {len(differences['sequences_created'])} sequence(s)")
    if differences["items_deleted"]:
        observation_parts.append(f"Deleted {len(differences['items_deleted'])} item(s)")
    if differences["sequences_updated"]:
        observation_parts.append(f"Updated {len(differences['sequences_updated'])} sequence(s)")
    if differences["items_renamed"]:
        observation_parts.append(f"Renamed {len(differences['items_renamed'])} item(s)")
    if differences["items_moved"]:
        observation_parts.append(f"Moved {len(differences['items_moved'])} item(s)")
    
    observation = ", ".join(observation_parts) if observation_parts else "No changes detected"

    
    return differences, observation

class EditProjectStrucure(BaseModel):
    prompt: str = Field(description="the prompt to edit the project structure")
    include_metadata :bool = Field(default=False, description="if True, the metadata will be included in the structure to help decision making. Slower")

@tool("edit_project_structure", args_schema=EditProjectStrucure)
async def edit_project_structure(prompt: str, include_metadata: bool = False):
    """
    Edit the project structure. Usefull to create Sequences, Bins, organize the project structure, etc...
    - usefull for update the settings of a sequence (as resolution, display format, etc...)
    - this tool can't change anything on a timeline
    - USE Metadata only if you need to know the transcription of audio, the music tempo, the description of the clips or images to perform the task
    """
    try:
        # Récupérer le contexte initial
        # final_context = await get_project_context(prompt, include_metadata)         

        initial_structure = await get_project_structure.ainvoke({"include_metadata": False})
        
        # Configuration de l'agent ReACT
        max_iterations = 20
        iteration = 0
        task_completed = False
        history = []
        
        # Boucle ReACT
        print(f"\n{'='*80}")
        while not task_completed and iteration < max_iterations:

            # Vérifier si l'utilisateur a demandé l'arrêt avant d'exécuter les actions
            if config.STOP_REQUESTED:
                print("\n🛑 Arrêt demandé par l'utilisateur pendant l'exécution des actions")
                config.API_STATUS = "End"
                return f"OBSERVATION: Task stopped by user during action execution at iteration {iteration}. Progress so far: {len(history)} action(s) completed."
            

            iteration += 1
            
            print(f"🔄 ITERATION {iteration}/{max_iterations}")

            

            config.API_STATUS = "Getting project structure..."
            current_structure = await get_project_structure.ainvoke({"include_metadata": False})
            
            # === PHASE 1: REASONING ======================================================
            config.API_STATUS = f"Thinking... (iteration {iteration})"
            
            # Construire le contexte avec l'historique
            history_text = ""
            if history:
                history_text = "\n\n### ACTIONS HISTORY:\n"
                for i, h in enumerate(history, 1):
                    history_text += f"\n**Iteration {i}:**\n"
                    history_text += f"- Reasoning: {h['reasoning']}\n"
                    history_text += f"- Actions performed: {len(h['actions'])} action(s)\n"
                    history_text += f"- Observation: {h['observation']}\n"
                    

            structured_output = {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your analysis and reasoning about what to do next"
                    },
                    "actions": {
                        "type": "array",
                        "description": "List of actions to perform (max 15 per batch). Use the tools defined in the tool_list.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the action/tool to call. Use the tools defined in the ACTION AVAILABLE."
                                },
                                "parameters": {
                                    "type": "string",
                                    "description": "The parameters for the action"
                                }
                            },
                            "required": ["name"]
                        }
                    }
                },
                "required": ["reasoning", "actions"]
            }    
            # Formater l'historique de manière propre
            # ### USER REQUESTS HISTORY:
            # {format_agent_history(config.AGENT_HISTORY)}
            reasoning_prompt = f"""

                You have performed the following actions before:
                ### AGENT HISTORY:
                {history_text}
                
                ### USER REQUEST:
                {prompt}

                ### CURRENT PROJECT STRUCTURE:
                {current_structure}

                ### YOUR TASK:
                Analyze the current state and decide the NEXT BATCH of actions to perform.

                ### RULES:
                1. ALWAYS and ONLY use an existing Id of an existing item
                2. You can only delete a bin if it is empty (no items inside); move items to an existing bin before deleting a bin
                3. you can only create a bin in an existing bin, so you can decompose you action in multiple tours if required in this specific case

                ### EXAMPLE RESPONSE:
                {json.dumps(structured_output, indent=2)}
                """.strip()
            
            system_instruction = f"""
                You are a professional video editing assistant using the ReACT framework.
                Your goal is to complete the user's request by planning and executing actions step by step.
                
                # ACTIONS AVAILABLE:
                {[EDIT_PROJECT_STRUCTURE_TOOL_LIST]}

                After each batch of actions, you will receive observations about what changed.
                Use this feedback to decide the next steps.

                You must follow this rules in your ReAct framework. 

                ### RULES for REASONING:
                - Be smart about batching: don't try to do everything at once. Break complex tasks into logical steps.
                - If MORE WORK is needed, generate the NEXT BATCH of actions (maximum 15 actions per batch)
                - Break down complex tasks into multiple iterations
                - Consider dependencies between actions
                            """.strip()



        
            # Appel au LLM pour le reasoning
            decision = gemini_call(reasoning_prompt, system_instruction, structured_output, None, 0.3, config.MODEL_AGENT_NAME)

            liste_actions = decision.get('actions', [])
            reasoning = decision.get('reasoning', 'No actions planned')
            
            print(f"💭 Reasoning: {reasoning}")

            # Cas 2 : Pas d'actions mais tâche non complétée = problème
            if not liste_actions:
                print("\n⚠️  No actions planned but task not completed. Ending...")
                return f"OBSERVATION: Unable to proceed. The agent couldn't determine any actions to perform. Reasoning: {reasoning}. Consider providing more details or checking if the request is feasible."
            

            
           # === PHASE 2: ACTING =========================================================
            
            action_prompt = f"""
                ### REASONING PHASE DECISION:
                {reasoning}
                
                ### ACTIONS TO EXECUTE:
                {liste_actions}
                
                """.strip()
            system_instruction = f"""
                You are a professional video editing assistant using the ReACT framework.
                Your task is to format the actions send by the Reasoning phase to the format expected by the JavaScript function.
                            """.strip()
            liste_actions = gemini_call(action_prompt, system_instruction, None, EDIT_PROJECT_STRUCTURE_TOOL_LIST, 0.3, MODEL_REACT_PROJECT_STRUCTURE_TOOL)
            
            config.API_STATUS = f"Executing actions... (iteration {iteration})"

            # Vérifier si l'utilisateur a demandé l'arrêt avant d'exécuter les actions
            if config.STOP_REQUESTED:
                print("\n🛑 Arrêt demandé par l'utilisateur pendant l'exécution des actions")
                config.API_STATUS = "End"
                return f"OBSERVATION: Task stopped by user during action execution at iteration {iteration}. Progress so far: {len(history)} action(s) completed."
            
            
            # Gestion du preset pour les séquences
            for action in liste_actions:
                print(action)
                if action.get("name") == "create_sequence":
                    try:
                        documents_preset_dir = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
                        documents_preset_path = documents_preset_dir / "PRESET_EDIT.sqpreset"
                        documents_preset_dir.mkdir(parents=True, exist_ok=True)
                        
                        if not documents_preset_path.exists():
                            project_root = Path(__file__).parent.parent.parent.parent
                            source_preset = project_root / "sequence" / "PRESET_EDIT.sqpreset"
                            if source_preset.exists():
                                shutil.copy2(source_preset, documents_preset_path)
                                print(f"  📦 Preset copied to: {documents_preset_path}")
                    except Exception as e:
                        print(f"  ⚠️  Error copying preset: {str(e)}")

                # décomposer en une série de modify item actions
                elif action.get("name") == "move_batch":
                    new_parent_nodeId = action.get("args", {}).get("target_bin_nodeId")


                    liste_nodeId_to_move  = []
                    start_nodeId = action.get("args", {}).get("start_item_nodeId")
                    end_nodeId = action.get("args", {}).get("end_item_nodeId")

                    def flatten_project_items(node):
                        items = []
                        if "children" in node:
                            for child in node["children"]:
                                items.append(child)
                                items.extend(flatten_project_items(child))
                        return items

                    flat_items = flatten_project_items(json.loads(current_structure))
                    add_item = False
                    for item in flat_items:
                        # print(item["nodeId"], start_nodeId, end_nodeId)
                        if item["nodeId"] == start_nodeId:
                            add_item = True
                        elif item["nodeId"] == end_nodeId:
                            liste_nodeId_to_move.append(item["nodeId"])
                            add_item = False
                            break

                        if add_item:
                            liste_nodeId_to_move.append(item["nodeId"])

                    
                    for nodeId in liste_nodeId_to_move:
                        liste_actions.append({
                            "name": "modify_item",
                            "args": {
                                "nodeId": nodeId,
                                "new_parent_nodeId": new_parent_nodeId
                            }
                        })
                        
            # Exécution des actions via JavaScript
            call_id = str(uuid.uuid4())
            script_arg = json.dumps(liste_actions)
            script = f"$._MYFUNCTIONS.executeActionList({script_arg});"
            
            PENDING_JS_CALLS[call_id] = {
                "args": {"script": script},
                "result": None,
                "status": "pending"
            }
            
            timeout = 180
            start_time = time.time()
            result = None
            while time.time() - start_time < timeout:
                if call_id in PENDING_JS_CALLS and PENDING_JS_CALLS[call_id]["status"] == "completed":
                    result = PENDING_JS_CALLS[call_id]["result"]
                    del PENDING_JS_CALLS[call_id]
                    break
                await asyncio.sleep(0.5)
            
            if call_id in PENDING_JS_CALLS:
                del PENDING_JS_CALLS[call_id]
            
            if result is None:
                print("❌ Timeout executing JavaScript actions")
                break
            

            
            # === PHASE 3: OBSERVATION ===
            config.API_STATUS = f"Observing changes... (iteration {iteration})"
            
            # Récupérer la structure après les actions
            time.sleep(1)
            post_action_structure = await get_project_structure.ainvoke({"include_metadata": False})
            differences, observation = compare_project_structures(current_structure, post_action_structure)
            
            print(f"📊 Observation: {observation}")
            
            # Ajouter à l'historique
            history.append({
                "reasoning": reasoning,
                "actions": liste_actions,
                "observation": observation,
                "differences": differences,
                "JSX_observation": result
            })



            # === PHASE 4: EST CE QUE LE USER PROMPT EST FINIT ?  =========================================================
            structured_output = {
                "type": "boolean",
                "description": "True if the user's request is fully completed and no more actions are needed"
            }

            system_instruction = """
                You are a professional video editor on Premiere Pro in a ReACT Agent framework. You are after the Reasoning and Acting phases.
                Your task is to decide if the USER PROMPT is fully completed. If True, the edit will stop here. If false, the agent will continue to perform the next batch of actions planned.
                You receive the USER PROMPT, the project structure before and after the actions, the list of actions performed and the observation of the actions.

                """.strip()

            final_prompt = f"""
                USER PROMPT: {prompt}
                PROJECT STRUCTURE BEFORE: {initial_structure}
                PROJECT STRUCTURE AFTER: {post_action_structure}
                LIST OF PERFORMED ACTIONS: {[iteration["actions"] for iteration in history]}
                OBSERVATION: {[iteration["JSX_observation"] for iteration in history]}
                """.strip()
            
            task_completed = gemini_call(final_prompt, system_instruction, structured_output, None, 0.2, MODEL_REACT_PROJECT_STRUCTURE)

            if task_completed:
                if iteration == 1 and not liste_actions:
                    return f"OBSERVATION: Task already completed. {reasoning}"
                break
             

        
        # Fin de la boucle ReACT

        print(f"Task completed: {task_completed}")
        print("="*80 + "\n")
        
        # Cas 3 : Max iterations atteintes sans complétion
        if not task_completed and iteration >= max_iterations:
            if history:
                summary = " | ".join([f"Iteration {i+1}: {h['reasoning']} - {h['observation']}" for i, h in enumerate(history)])
                return f"OBSERVATION: Maximum iterations ({max_iterations}) reached. Task partially completed. Actions performed: {summary}. Consider breaking down the task or continuing manually."
            else:
                return f"OBSERVATION: Maximum iterations ({max_iterations}) reached without completing any actions. The task might be too complex or unclear."
        

        
        # Génération du résultat final pour succès normal
        final_summary_parts = []
        for i, h in enumerate(history, 1):
            final_summary_parts.append(f"Iteration {i}: {h['reasoning']} - {h['observation']}")
        
        final_result = "OBSERVATION: Project structure edited successfully. " + " | ".join(final_summary_parts)
        
        return final_result
        
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            log_lignes = []
            for trace in tb:
                log_lignes.append(f"Fichier \"{trace.filename}\", ligne {trace.lineno}, dans {trace.name}")
            log_complet = "\n".join(log_lignes)
            message = f"Erreur edit_project_structure : {str(e)}\nTraceback complet :\n{log_complet}"
        else:
            message = f"Erreur edit_project_structure: {str(e)}"
        print(message)
        return "Error: " + message

class AudioTypeEnum(str, Enum):
    music = "music"
    speech = "speech"

class LabelizeAudio(BaseModel):
    audioType: AudioTypeEnum = Field(description="The type of the audio to labelize, can be 'music' or 'speech'")
    audioName: str = Field(description="The audio name of the audio file in the project to labelize")

@tool("labelize_audio", args_schema=LabelizeAudio)
async def labelize_audio(audioType: str, audioName: str):
    """
    Labelize an audio file
    Usefull to get music downbeats or transcription of audio
    - ASK the user to provide the name of the audio file and the type (music or speech) if not given
    """
    
    # 1. Récupération du path depuis la structure du projet & Vérifier si l'audio est déjà labelisé
    try : 
        project_V0 = await get_project_structure.ainvoke({"include_metadata": False, "skip_labelize": True})
        project_V0 = json.loads(project_V0)
        
        
        # Récupération du nodeId
        system_instruction = f"""
        You are a professional video editor expert in Premiere Pro.
        You receive an file Name.
        Your task is to find the nodeId of the file in the project: 
        ### PROJECT 

        {project_V0}
        
        """
        
        structured_output = {
            "type": "object",
            "properties": {
                "nodeId": {"type": "string", "description": "The nodeId of the audio file in the project"}
            },
            "required": ["nodeId"],
        }
        
        nodeId = gemini_call(audioName, system_instruction, structured_output, None, 0.2, "gemini-2.5-flash-lite" )['nodeId']

        
        project_V0 = await get_project_structure.ainvoke({"include_metadata": True, "skip_labelize": True})
        project_V0 = json.loads(project_V0)
        
        
        path_clip_db = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "index.json"
        with open(path_clip_db, 'r', encoding='utf-8') as f:
            clip_db = json.load(f)
        
        
        # Fonction récursive pour chercher dans tous les niveaux
        def find_item_by_nodeid(items, target_nodeid):
            for item in items:
                if item.get("nodeId") == target_nodeid:
                    return item
                if "children" in item:
                    result = find_item_by_nodeid(item["children"], target_nodeid)
                    if result:
                        return result
            return None
        
        clip_found = None
        audio_path = None
        
        # Chercher l'item dans toute l'arborescence
        found_item = find_item_by_nodeid(project_V0.get("children", []), nodeId)
        
        if found_item:
            audio_path = found_item.get("mediaPath")
            
            # Si l'audio Path est un mp4 : il faut le transcrire en mp3
            #@TODO : a raccorder
            
            # chercher dans index si on l'a déjà labelisé ou pas
            if audio_path:
                for item in clip_db:
                    if item["mediaPath"] == audio_path:
                        clip_found = item
                        if audioType == "music":
                            if "downbeats" in item or "beats" in item:
                                return "Music already labeled"

                        elif audioType == "speech":
                            if "transcription" in item:
                                return "Speech already labeled"
                        break
        
        # Vérifier si l'audio a été trouvé
        if audio_path is None:
            return f"Error: Audio file '{audioName}' with nodeId '{nodeId}' not found in project" 


        # 2. Labelization
        if audioType == "music":
            downbeats, beats, json_path = analyze_music(audio_path)
        elif audioType == "speech":
            
            # 2.1. Récupération du path de l'audio
            call_id = str(uuid.uuid4())
            script = f"$._MYFUNCTIONS.AGENT_SPEECH_labelizeAudio('{nodeId}');"
            PENDING_JS_CALLS[call_id] = {
                "args": {"script": script},
                "result": None,
                "status": "pending"
            }
            timeout = 30
            start_time = time.time()
            result = None
            while time.time() - start_time < timeout:
                if call_id in PENDING_JS_CALLS and PENDING_JS_CALLS[call_id]["status"] == "completed":
                    result = PENDING_JS_CALLS[call_id]["result"]
                    del PENDING_JS_CALLS[call_id]
                    break
                await asyncio.sleep(0.5)
            if call_id in PENDING_JS_CALLS:
                del PENDING_JS_CALLS[call_id]
            if result is None:
                return "Error: Timed out waiting for JavaScript function to execute."
            audio_path = result
            transcription = main_transcription_for_agent(audio_path, config.AGENT_TOKEN) 
        # print(f"Transcription: {transcription}")

        # 3. Ajout aux metadata du projet
        for item in project_V0["children"]:
            if item["nodeId"] == nodeId:
                if audioType == "music":
                    item["downbeats"] = downbeats
                    item["beats"] = beats
                    item["json_path"] = json_path
                elif audioType == "speech":
                    item["transcription"] = transcription
                    item["json_path"] = json_path
                break
        

        # 4. Sauvegarde du projet
        base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
        Path(base_path).mkdir(parents=True, exist_ok=True)
        project_id = project_V0.get("projectID", "")
        file_path = base_path / f"{project_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(project_V0, f, indent=2, ensure_ascii=False)
        
        
        # 5. Sauvegarde de la base de données INDEX


        if audioType == "music":
            if clip_found is not None:
                clip_found["downbeats"] = downbeats
                # clip_found["beats"] = beats
                clip_found["json_path"] = json_path
            else:
                clip_db.append({
                    "mediaPath": audio_path,
                    "downbeats": downbeats,
                    # "beats": beats,
                    "json_path": json_path
                })
        elif audioType == "speech":
            if clip_found is not None:
                clip_found["transcription"] = transcription
            else:
                clip_db.append({
                "mediaPath": audio_path,
                "transcription": transcription,
            })

        with open(path_clip_db, 'w', encoding='utf-8') as f:
            json.dump(clip_db, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erreur labelize_audio: {str(e)}")
        return "Error: " + str(e)
    return "done"



