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
from LIB.config import PENDING_JS_CALLS
from LIB.music_analysis_V2 import analyze_music
from LIB.subtitles import main_transcription_for_agent
from LIB.custom_llm import PremiereGPT_LLM

AGENT_TOKEN = None

MODEL_CONTEXT_1 = "gemini-2.5-flash-lite" # Get Project Context
MODEL_CONTEXT_2 = "gemini-2.5-flash-lite" # Get Timeline Context
MODEL_TOOL_1 = "gemini-2.5-flash-lite" # Edit project structure
MODEL_TOOL_2 = "gemini-2.5-flash-lite" # Choose Effect properties



TOOL_DISPLAY_MAPPING = {
    "get_project_structure": {
        "title": "Grepping project",
        "category": "extendscript"
    },
    "edit_project_structure": {
        "title": "Editing project",
        "category": "extendscript"
    },
    "get_timeline_structure": {
        "title": "Grepping timeline",
        "category": "extendscript"
    },
    "edit_timeline_structure": {
        "title": "Editing timeline",
        "category": "extendscript"
    }, 
    "labelize_audio": {
        "title": "Grepping audio",
        "category": "extendscript"
    },
    "open_timeline": {
        "title": "Opening timeline",
        "category": "extendscript"
    }, 
    "get_creative_todo": {
        "title": "Crafting todo",
        "category": "extendscript"
    }
}


def gemini_call(prompt, system_instruction, structured_output = None, tool_list = None, temperature= 0.1,  model = "gemini-2.5-flash-lite"): 
    """
    Appelle l'API Gemini via l'API relai.
    Conserve la même interface et le même comportement que l'ancienne fonction.
    """
    global AGENT_TOKEN
    
    
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
            headers={"Authorization": f"Bearer {AGENT_TOKEN}"}
        )
        
        # Vérifier le statut de la réponse
        if response.status_code != 200:
            raise Exception(f"Erreur API (status {response.status_code}): {response.text}")
        
        # Extraire le résultat
        result = response.json().get("result")
        
        # Retourner le résultat dans le même format que l'ancienne fonction
        return result
        
    except requests.exceptions.RequestException as e:
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
                "Authorization": f"Bearer {AGENT_TOKEN}",
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
    include_metadata :bool = Field(default=False, description="if True, the metadata will be included in the structure to help decision making. Slower")

@tool("get_project_structure", args_schema=GetProjectStructure)
async def get_project_structure(include_metadata: bool = False, skip_labelize: bool = False):
    """
    Returns the JSON structure of the Premiere Pro project
    Usefull ot get the availables clips, musics, audio, folders
    - USE Metadata only if you need to know the transcription of audio, the music tempo, the description of the clips or images to perform the task

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
        print(result)
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



        
        
        # 3. On récupère tous les paths des fichiers images et vidéos
        def extract_media_paths(item, paths_list):
            """Parcourt récursivement un élément et extrait tous les paths des médias vidéo et image"""
            if isinstance(item, dict):
                # Vérifier si l'élément actuel est une vidéo ou une image
                item_type = item.get('type', '')
                if item_type in ['Video', 'Image'] and 'mediaPath' in item:
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
            print(f"Fichier index.json créé : {path_clip_db}")
        
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
        # if include_metadata:
        #     # Recharger la base de données car elle a pu être mise à jour
        #     with open(path_clip_db, 'r', encoding='utf-8') as f:
        #         clip_db_updated = json.load(f)
            
        #     metadata_map = {item['mediaPath']: item for item in clip_db_updated}

        #     def add_metadata_to_project_structure(item):
        #         """Parcourt récursivement la structure du projet pour y ajouter les métadonnées."""
        #         if isinstance(item, dict):
        #             item_type = item.get('type', '')
        #             if item_type in ['Video', 'Image'] and 'mediaPath' in item:
        #                 media_path = item['mediaPath']
        #                 if media_path in metadata_map:
        #                     # On ne garde que les clés de métadonnées pertinentes
        #                     meta = metadata_map[media_path]
        #                     item['metadata'] = {
        #                         "description": meta.get("description"),
        #                         "camera_angle": meta.get("camera_angle"),
        #                         "colors": meta.get("colors"),
        #                         "people": meta.get("people")
        #                     }
                    
        #             if 'children' in item and isinstance(item['children'], list):
        #                 for child in item['children']:
        #                     add_metadata_to_project_structure(child)
        #         elif isinstance(item, list):
        #             for sub_item in item:
        #                 add_metadata_to_project_structure(sub_item)

        #     add_metadata_to_project_structure(project_data)
            
            # Le résultat final est la structure du projet mise à jour et convertie en JSON
        #     final_result = json.dumps(project_data, indent=2, ensure_ascii=False)
        # else:
        #     # Sinon, on retourne le JSON original
        final_result = result

        
        config.API_STATUS = "Thinking..."
        
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
    try : 
        

            
        # final_context = await get_project_context(prompt, include_metadata) 
        final_context = await get_project_structure.ainvoke({"include_metadata": include_metadata})
    
        # 4. Génération des opérations à effectuer
        print("EDIT PROJECT STRUCTURE: Final prompt:", prompt)
        full_prompt = f"""    
        ### Project Architecture context :
        {final_context}
        
        ### User prompt : 
        {prompt}
            """.strip()
            
        system_instruction="""
    You are a professional video editor. You receive a JSON representation of some items in the project. 
    Your task is to generate the list of the operations to perform the user prompt. 
    Keep in mind that this list will be executed in the right order, so you must consider the dependencies between the operations.

    ### TASKS:
    - Analyse the context and the user prompt
    - Use the context if needed. 
    - Generate the list of the operations in the right order to perform the user prompt
            """, 
            

        create_bin_tool = {
            "name": "create_bin",
            "description": """
            Create a new bin in the project at the right place. A bin is also called 'folder', 'dossier' or 'bin'
            * create a bin at the root (using "/" as parent path) if the user prompt don't specify a parent path
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the bin"
                    },
                    "parent_path": {
                        "type": "string",
                        "description": "The path of the parent bin",
                    },
                },
                "required": ["name", "parent_path"],
            },
        }
        
        delete_bin_tool = {
            "name": "delete_bin",
            "description": "Delete a bin in the project. works only with bins",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodeId": {
                        "type": "string",
                        "description": "The nodeId of the bin to delete",
                    },
                },
                "required": ["nodeId"],
            },
        }

        create_sequence_tool = {
            "name": "create_sequence",
            "description": """
            Create a new sequence in the project at the right place. 
            
            ### RULES:
            * Usually, the prompt start with width and height of the sequence
            * Always respect the width and height of the sequence if given
            * If width and height are not given, choose a STANDARD RESOLUTION based on the purpose of the prompt.  
            
            ### STANDARD RESOLUTIONS:
            - Tik Tok, Insta Reels, Shorts : width 1080, height 1920
            - Youtube : width 1920, height 1080
            - Film : width 3840, height 2160
            - Film Cinematic : width 3840, height 1634
                """,
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the sequence"
                    },
                    "parent_path": {
                        "type": "string",
                        "description": "The path of the parent bin",
                    },
                    "videoFrameWidth": {
                        "type": "number",
                        "description": "The width of the sequence",
                    },
                    "videoFrameHeight": {
                        "type": "number",
                        "description": "The height of the sequence",
                    },
                    "videoDisplayFormat": {
                        "type": "string",
                        "description": "The display format of the sequence",
                        "enum": ["23.976fps", "24fps", "25fps", "30fps", "50fps", "60fps"]
                    },
                },
                "required": ["name", "parent_path", "videoFrameHeight", "videoFrameWidth", "videoDisplayFormat"],
            },
        }

        update_sequence_tool = {
            "name": "update_sequence",
            "description": "Update the settings of a sequence in the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodeId": {
                        "type": "string",
                        "description": "The nodeId of the sequence to update",
                    },
                    "videoFrameHeight": {
                        "type": "number",
                        "description": "The height of the sequence",
                    },
                    "videoFrameWidth": {
                        "type": "number",
                        "description": "The width of the sequence",
                    },
                    "videoDisplayFormat": {
                        "type": "string",
                        "description": "The display format of the sequence",
                        "enum": ["23.976fps", "24fps", "25fps", "29.97fps", "30fps", "48fps", "50fps", "59.94fps", "60fps"]
                    },
                },
                "required": ["nodeId", "videoFrameHeight", "videoFrameWidth", "videoDisplayFormat"],
            },
        }
        
        clone_sequence_tool = {
            "name": "duplicate_sequence",
            "description": "Clone a sequence in the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodeId": {
                        "type": "string",
                        "description": "The nodeId of the sequence to duplicate",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "The name of the new sequence",
                    },
                },
                "required": ["nodeId", "new_name"],
            },
        }

        modify_item_tool = {
            "name": "modify_item",
            "description": """Modify an item in the project,
            - if new_name is provided, it will rename the item
            - if new_parent_path is provided, it will move the item to the new path
            - if it's a bin it will move all the items inside, usefull to move many items
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "nodeId": {
                        "type": "string",
                        "description": "The nodeId of the item to move",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "The new name of the item",
                    },
                    "new_parent_path": {
                        "type": "string",
                        "description": "The path of the new parent bin",
                    },
                },
                "required": ["nodeId"],
            },
        }

        # find_item_tool = {
        #     "name": "find_item",
        #     "description": "Find an item in the project",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "nodeId": {
        #                 "type": "string",
        #                 "description": "The nodeId of the item to find",
        #             },
        #         },
        #         "required": ["nodeId"],
        #     },
        # }

        
        tool_list = [create_bin_tool, delete_bin_tool, 
                    create_sequence_tool, update_sequence_tool, clone_sequence_tool, 
                    modify_item_tool]

        
        config.API_STATUS = "Thinking..."
        liste_actions = gemini_call(full_prompt, system_instruction, None, tool_list, 0.3, MODEL_TOOL_1 )
    except Exception as e:
        print(f"Erreur edit_project_structure: {str(e)}")
        return "Error: " + str(e)
    
    print("--- [ACTION CALL PROJECT] --- " + str(liste_actions))

    # 4.5. Vérifier si une action de création de séquence est présente et copier le preset si nécessaire

    for action in liste_actions:
        if action.get("name") == "create_sequence":

            try:
                # Définir le chemin de destination du preset
                documents_preset_dir = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
                documents_preset_path = documents_preset_dir / "PRESET_EDIT.sqpreset"
                
                # Créer le dossier s'il n'existe pas
                documents_preset_dir.mkdir(parents=True, exist_ok=True)
                print("Documents preset dir created")
                # Copier le preset s'il n'existe pas déjà
                if not documents_preset_path.exists():
                    # Trouver le preset source - remonter à la racine du projet (arctan-22/)
                    project_root = Path(__file__).parent.parent
                    print(f"Project root: {project_root}")
                    
                    # Chemin dans la version de développement
                    source_preset = project_root / "sequence" / "PRESET_EDIT.sqpreset"
                    

                    if source_preset.exists():
                        shutil.copy2(source_preset, documents_preset_path)
                        print(f"Preset copié vers: {documents_preset_path}")
                    else:
                        print(f"Attention: Preset source non trouvé à {source_preset}")
                
            except Exception as e:
                print(f"Erreur lors de la copie du preset: {str(e)}")
            
            # Une seule copie suffit, on sort de la boucle
            break

    # 5. Application des opérations à effectuer
    config.API_STATUS = "Performing actions..."
    call_id = str(uuid.uuid4())
    
    script_arg = json.dumps(liste_actions)
    script = f"$._MYFUNCTIONS.executeActionList({script_arg});"

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
    
    

    # Résumé pour dire a l'agent que la fonction a été executée
    
    # result = f"OBSERVATION : The project structure has been updated successfully. "    
    # result = f"OBSERVATION : tool edit_project_structure has been executed successfully. The following actions have been performed: {liste_actions}"    
    # result = f"OBSERVATION : tool edit_project_structure has been executed successfully this prompt: ## PROMPT ## {prompt} ."    
    result = f"OBSERVATION : The project structure has been edited. The following actions have been performed: {liste_actions}"    


    return result

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
            #@TODO : Transcrire en mp3
            
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
            transcription = main_transcription_for_agent(audio_path, AGENT_TOKEN) # a racorder
        # print(f"Transcription: {transcription}")

        # 3. Ajout aux metadata du projet
        # for item in project_V0["children"]:
        #     if item["nodeId"] == nodeId:
        #         if audioType == "music":
        #             item["downbeats"] = downbeats
        #             item["beats"] = beats
        #             item["json_path"] = json_path
        #         elif audioType == "speech":
        #             item["transcription"] = transcription
        #             item["json_path"] = json_path
        #         break
        

        # 4. Sauvegarde du projet
        # base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
        # Path(base_path).mkdir(parents=True, exist_ok=True)
        # project_id = project_V0.get("projectID", "")
        # file_path = base_path / f"{project_id}.json"
        # with open(file_path, 'w', encoding='utf-8') as f:
        #     json.dump(project_V0, f, indent=2, ensure_ascii=False)
        
        
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





# ------------- TIMELINE MANAGEMENT -------------

async def apply_effect(ID, prompt, piste):
    """
    Détermine :
    - le nom de l'effet
    - si on doit le créer ou modifier
    - le nom des propriétés à modifier, et leurs valeurs à modifier
    - les éventuelles keyframes
    """
    
    #1. Récupérer le nom de l'effet, ses propriétés
    effects = [
        {
            "name": "AE.ADBE Lumetri",
            "description": "to perform color correction",
            14 : "Temperature, range -300 - 300 (initial value 0)",
            15 : "Tint, range 300 - 300 (initial value 0)",
            16 : "Saturation, range 0 - 300 (initial value 100)",
            19 : "Exposure, range -7 - 7 (initial value 0)",
            20 : "Contrast, range -150 - 150 (initial value 0)",
            21 : "Highlights, range -150 - 150 (initial value 0)",
            22 : "Shadows, range -150 - 150 (initial value 0)",
            23 : "Whites, range -150 - 150 (initial value 100)",
            24 : "Blacks, range -150 - 150 (initial value 0)",
        },
        {
            "name": "AE.ADBE Opacity",
            "description": "to perform opacity",
            0: "Opacity, range 0-100",
        },
        {
            "name": "AE.ADBE Motion",
            "description": "to perform motion, position",
            0: "Position X, range -0.5 - 1.5  (initial centered value 0.5)",
            1: "Position Y, range -0.5 - 1.5  (initial centered value 0.5)", # merge avec la 0 pour former un array [x, y]
            2: "Scale, range 0-1000 (initial value 100)",
            4: "Rotation, range 0-360 (initial value 0)",
            5: "Anchor Point X, range -0.5 - 1.5  (initial centered value 0.5)",
            6: "Anchor Point Y, range -0.5 - 1.5  (initial centered value 0.5)", # merge avec la 5 pour former un array [x, y]           
            7 : "Crop Left, range 0-100 (initial value 0)",
            8 : "Crop Top, range 0-100 (initial value 0)",
            9 : "Crop Right, range 0-100 (initial value 0)",
            10 : "Crop Bottom, range 0-100 (initial value 0)",
            11 : "Anti-flicker Filter, range 0-1 (initial value 0)", # remettre en 6 
        }
    ]
    
    
    try: 

        # 1.1 Choix des noms des effets
        effect_list = [{"name": effect["name"], "description": effect["description"]} for effect in effects]
        
        full_prompt = f"""    
        ### USER PROMPT : 
        {prompt}
        """.strip()
            
        project_effect_instruction=f"""
        You are a professional video editor expert in Premiere Pro.
        Your task is to select the right effects to perform the user prompt.

        ### RULES:

        ### EFFECTS:
        {effect_list}
        """
        
        effect_names_enum = [effect["name"] for effect in effects]
        effects_structured_output = { 
            "type": "array", 
            "items": { 
                "type": "string", 
                "description": "The name of the effect to apply",
                "enum": effect_names_enum
            }
        }
        effects_to_apply = gemini_call(full_prompt, project_effect_instruction, effects_structured_output, None, 0.2, "gemini-2.5-flash-lite" )

        # 1.2 Choix des propriétés & valeurs
        effects_to_apply_with_properties = [effect for effect in effects if effect["name"] in effects_to_apply]
        
        full_prompt = f"""    
        ### USER PROMPT : 
        {prompt}
        """.strip()
        
        project_effect_instruction=f"""
        You are a professional video editor expert in Premiere Pro.
        Your task is to select the right properties of these effects to perform the user prompt.

        ### RULES:
        - only return the properties that are needed t be edited to perform the user prompt
        - always respect the range of the property values to edit
        - if the prompt solo ask to add an effect, return 999 as property number and a single property to edit

        ### EFFECTS:
        {effects_to_apply_with_properties}
        """
        
        effects_properties_structured_output = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "effect_name": {
                        "type": "string",
                        "description": "The name of the effect to apply",
                        "enum": [effect["name"] for effect in effects_to_apply_with_properties],
                    },
                    "property_number": {
                        "type": "number",
                        "description": "The number of the property to apply",
                    },
                    "property_value": {
                        "type": "number",
                        "description": "The value of the property to apply",
                    }
                },
                "required": ["effect_name", "property_number", "property_value"],
            }
        }

        
        effects_properties = gemini_call(full_prompt, project_effect_instruction, effects_properties_structured_output, None, 0.5, MODEL_TOOL_2 )
        
        print(json.dumps(effects_properties, indent=2))
        
        
        # 2. Appliquer les effets
        type_piste = piste.split(" ")[0]
        no_piste = int(piste.split(" ")[1])
        for effect_property in effects_properties:
            args =  {
                        "ID": ID,
                        "effect_name": effect_property["effect_name"],
                        "property_number": effect_property["property_number"],
                        "property_value": effect_property["property_value"],
                        "type_piste": type_piste,
                        "no_piste": no_piste
                    }
            
            # Appeler la fonction ExtendScript
            call_id = str(uuid.uuid4())
            script_arg = json.dumps(args)
            script = f"$._MYFUNCTIONS.addEffect({script_arg});"
            
            PENDING_JS_CALLS[call_id] = {
                "args": {"script": script},
                "result": None,
                "status": "pending"
            }
            
            # Attendre la réponse
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
                print(f"⚠️ Timeout lors de l'application de l'effet {effect_property['effect_name']}")
            else:
                print(f"✅ Effet {effect_property['effect_name']} appliqué avec succès")
            

        
    
    except Exception as e:
        print(f"Erreur apply_effect: {str(e)}")
        return "Error: " + str(e)
    
    return "done"

def find_item_by_node_id(items, node_id):
    for item in items:
        if item.get("nodeId") == node_id:
            return item
        if "children" in item and item["children"]:
            found = find_item_by_node_id(item["children"], node_id)
            if found:
                return found
    return None

class OpenTimeline(BaseModel):
    timeline_name: str = Field(description="The name of the timeline to open")

@tool("open_timeline", args_schema=OpenTimeline)
async def open_timeline(timeline_name: str):
    """
    Open a timeline in Premiere Pro by its name
    Usefull to open a timeline before editing it or grepping it
    
    """
    try:
        # 1. Récupération du nodeId de la timeline
        
        project_V0 = await get_project_structure.ainvoke({"include_metadata": False})
        project_V0 = json.loads(project_V0)
        
        system_instruction = f"""
        You are a professional video editor. You receive a JSON representation of the project architecture.
        Your task is to select the right timeline to open.

        ### TASKS:
        - Analyse the project architecture and the user prompt
        - Select the right timeline to open
        - Return the nodeId of the timeline
        
        ### PROJECT
        {project_V0}

        """
        
        
        structured_output = {
            "type": "object",
            "properties": {
                "nodeId": {"type": "string", "description": "The nodeId of the timeline to open"}
            },
            "required": ["nodeId"],
        }
        
        nodeId = gemini_call(timeline_name, system_instruction, structured_output, None, 0.2, "gemini-2.5-flash-lite" )['nodeId']
        
        
        # 2. Appel du script pour ouvrir la timeline
        call_id = str(uuid.uuid4())
        
        script = f"$._MYFUNCTIONS.activateSequenceById('{nodeId}');"
        
        PENDING_JS_CALLS[call_id] = {
            "args": {"script": script},
            "result": None,
            "status": "pending"
        }
        
        # 3. Attendre le résultat de l'exécution
        result = None
        timeout = 10  # 10 secondes de timeout
        start_time = time.time()
        
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
        
        
        result = f"OBSERVATION : Timeline {timeline_name} has been opened successfully."
        return result  
        
    except Exception as e:
        print(f"Erreur open_timeline: {str(e)}")
        return "Error: " + str(e)
    
class GetTimelineStructure(BaseModel):
    include_metadata :bool = Field(default=False, description="if True, the metadata will be included in the structure to help decision making. Slower")
    include_effects :bool = Field(default=False, description="if True, the effects will be included in the structure. Slower")
    
@tool("get_timeline_structure", args_schema=GetTimelineStructure)
async def get_timeline_structure(include_metadata: bool = False, include_effects: bool = False):
    """
    Returns the JSON structure of the active timeline in Premiere Pro
    Usefull to read the timeline informations
    - USE open_timeline to open the timeline before getting the structure if needed. 
    - USE Metadata only if you need to know the transcription of audio, the music tempo, the description of the clips or images to perform the task
    - USE Effects only if you need to know the effects of the clips to perform the task
    """
    call_id = str(uuid.uuid4())
    
    config.API_STATUS = "Getting timeline structure..."
    
    # 1. Récupération de la structure de la timeline
    PENDING_JS_CALLS[call_id] = {
        "args": {"script": "$._MYFUNCTIONS.getSequenceStructure();"},
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

    # print(result)
    result = json.loads(result)
    
    # 1. Chargement des effets
    if not include_effects:
        for clip in result:
            clip.pop('effects')
            
    
    # 2. Chargement des métadonnées
    if include_metadata:
        
        project_V0 = await get_project_structure.ainvoke({"include_metadata": include_metadata})
        project_V0 = json.loads(project_V0)



        for clip in result:
            if "nodeId" in clip and clip["nodeId"]:
                project_item = find_item_by_node_id(project_V0.get("children", []), clip["nodeId"])
                if project_item and "metadata" in project_item:
                    clip["metadata"] = project_item["metadata"]
                    
                    
    # 3. supprimer les item audio qui sont liés a une video et qui n'ont pas de metadata
    video_items = [item for item in result if 'video' in item.get('piste', '')]
    audio_items = [item for item in result if 'audio' in item.get('piste', '')]

    # Pour comparer les metadonnées (dictionnaires), on les convertit en chaînes JSON triées
    def get_metadata_str(item):
        # Utiliser json.dumps avec sort_keys=True garantit une représentation textuelle unique
        return json.dumps(item.get('metadata', {}), sort_keys=True)

    # Créer un ensemble de "signatures" uniques pour les éléments vidéo pour une recherche rapide
    # Une signature est un tuple (nodeId, start, end, metadata_string)
    video_signatures = {
        (item['nodeId'], item['start'], item['end'], get_metadata_str(item))
        for item in video_items
    }

    # Conserver uniquement les éléments audio qui n'ont pas de doublon vidéo correspondant
    audio_items_uniques = []
    for item in audio_items:
        audio_signature = (item['nodeId'], item['start'], item['end'], get_metadata_str(item))
        if audio_signature not in video_signatures:
            audio_items_uniques.append(item)

    # Le résultat final est la combinaison de tous les éléments vidéo et des audios uniques
    result = video_items + audio_items_uniques
                    
    
    print(json.dumps(result, indent=2))

    return result

async def get_timeline_context(prompt: str, include_metadata: bool = False, include_effects: bool = False):
    
    config.API_STATUS = "Getting timeline context..."
    
    timeline_V0 = await get_timeline_structure.ainvoke({"include_metadata": include_metadata, "include_effects": include_effects})
        
    full_prompt = f"""    
    ### JSON timeline :
    {timeline_V0}
    
    ### User prompt : 
    {prompt}
        """.strip()

    system_instruction="""
    You are a professional video editor. You receive a JSON representation of the timeline.
    Your task is to select the right context that will be required to perform the user prompt.

    ### TASKS:
    - Analyse the timeline and the user prompt
    - Select the rights time windows (start, end) to add to the context that will be required to perform the user prompt
    - all the item starting in this time windows will be added to the context
    - Return the list of the time windows (start, end)

    ### RULES:
    - if no context is needed, return an empty list
    - if all the context is needed, return a unique item with the first start and the last end
        """, 

    structured_output = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "number",
                    "description": "The start of the time window"
                },
                "end": {
                    "type": "number",
                    "description": "The end of the time window"
                }
            },
            "required": ["start", "end"]
        }
    }
            
    context = gemini_call(full_prompt, system_instruction, structured_output, None, 0.2, MODEL_CONTEXT_2 )
    

    final_context = []
    if context:
        # Assurer que timeline_V0 est bien un dictionnaire Python
        timeline_data = json.loads(timeline_V0) if isinstance(timeline_V0, str) else timeline_V0
        
        for time_window in context:
            start_window = time_window['start']
            end_window = time_window['end']
            
            # Filtrer les items de la timeline
            for item in timeline_data:
                if start_window <= item.get('start', -1) <= end_window:
                    # Éviter les doublons si un item est dans plusieurs fenêtres
                    if item not in final_context:
                        final_context.append(item)
    
    
    
    

class EditTimelineStructure(BaseModel):
    prompt: str = Field(description="the prompt to edit the timeline structure")
    include_metadata :bool = Field(default=False, description="if True, the metadata will be included in the structure to help decision making. Slower")
    include_effects :bool = Field(default=False, description="if True, the effects will be included in the structure. Slower")
    
@tool("edit_timeline_structure", args_schema=EditTimelineStructure)
async def edit_timeline_structure(prompt: str, include_metadata: bool = False, include_effects: bool = False):
    """
    Edit the Premiere Pro active sequence (also call timeline). 
    Usefull to add clips, audio, effects, edit etc...
    
    ### METADATA: 
    if metadata is False, you will only have the name, settings, resolution of the item
    if metadata is True, you will have description of the item, what is seen of the image, the transcription of audio, the music tempo
    
    - USE metadata if the user describe the content of the clips or images to perform the task
    - USE Metadata only if you need to know the transcription of audio, the music tempo, the description of the clips or images to perform the task
    - USE include effect if your task is related to effects
    """
    
    try: 
    
        project_final_context = await get_project_context(prompt, include_metadata)
        final_context = await get_timeline_context(prompt, include_metadata, include_effects)
        
        
        #3. Générer la liste d'action 
        
        move_item_tool = {
            "name": "move_item",
            "description": """
            Move an item at a precise position in the timeline
            - use end to precise the duration of the item
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "ID": {
                        "type": "string",
                        "description": "The ID in the timeline of the item to move"
                    },
                    "track_number": {
                        "type": "number",
                        "description": "The track number to move the item to. starting from 1",
                    },
                    "start": {
                        "type": "number",
                        "description": "The start to move the item to",
                    },
                    "end": {
                        "type": "number",
                        "description": "The end to move the item to",
                    },
                    "ripple": {
                        "type": "boolean",
                        "description": "If True, the item will ripple the existing items",
                        "default": False,
                    }
                },
                "required": ["ID", "start"],
            },
        }
        
        insert_item_tool = {
            "name": "insert_item",
            "description": """
            Insert an item at a precise position in the timeline
            - use start to precise the position to insert the item
            - use end to precise the duration of the item
            - use inPoint_source and outPoint_source to precise the part of the item to insert
            
            ### RULES:
            - if not precised, insert at the end of the timeline
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "nodeId": {
                        "type": "string",
                        "description": "The nodeId in the project of the item to insert",
                    },
                    # "type": {
                    #     "type": "string",
                    #     "description": "The type of the item to move. video or audio",
                    #     "enum": ["video", "audio"],
                    # },
                    "track_index": {
                        "type": "number",
                        "description": "The track index to insert the item to. starting from 0",
                    },
                    "start": {
                        "type": "number",
                        "description": "The start to insert the item to",
                    },
                    "end": {
                        "type": "number",
                        "description": "The end to insert the item to",
                    },
                    "inPoint_source": {
                        "type": "number",
                        "description": "The inPoint of the item to insert. usefull to insert a specific part of the item, as music, or speech",
                    },
                    "outPoint_source": {
                        "type": "number",
                        "description": "The outPoint of the item to insert. usefull to insert a specific part of the item, as music, or speech",
                    },
                    "ripple": {
                        "type": "boolean",
                        "description": "If True, the item will ripple the existing items",
                        "default": False,
                    },
                },
                "required": ["nodeId", "start"],
            },
        }
        
        delete_item_tool = {
            "name": "delete_item",
            "description": "Delete an item in the timeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "ID": {
                        "type": "string",
                        "description": "The ID on the timeline of the item to delete",
                    },
                    "ripple": {
                        "type": "boolean",
                        "description": "If True, the item will ripple the existing items. Most of the time, it's False",
                        "default": False,
                    },
                },
                "required": ["ID"],
            },
        }
        
        add_marker_tool = {
            "name": "add_marker",
            "description": """Add a marker to the timeline at a specific time.
            Markers can be used to mark important points, chapters, or sections.
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "number",
                        "description": "The time in seconds where the marker should be created",
                    },
                    "comments": {
                        "type": "string",
                        "description": "The comment/name of the marker",
                    },
                    "type": {
                        "type": "string",
                        "description": "The type of marker",
                        "enum": ["Comment", "Chapter", "Segmentation", "WebLink"],
                        "default": "Comment"
                    },
                    "color": {
                        "type": "integer",
                        "description": "Color index (0-7). 0=Green, 1=Red, 2=Purple, 3=Orange, 4=Yellow, 5=White, 6=Blue, 7=Cyan",
                        "minimum": 0,
                        "maximum": 7,
                        "default": 0
                    },
                    "end": {
                        "type": "number",
                        "description": "Optional end time in seconds if the marker spans a duration (must be greater than time)",
                    },
                },
                "required": ["time", "comments"],
            },
        }

        edit_effect_tool = {
            "name": "edit_effect",
            "description": """
            Apply an effect to items in the timeline
            - use the prompt to describe the desired effect
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "effects": {
                        "type": "array",
                        "description": "List of effects to apply",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ID": {
                                    "type": "string",
                                    "description": "The id of the item to apply the effect to",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "The prompt to edit the effect",
                                },
                            },
                            "required": ["ID", "prompt"],
                        }
                    }
                },
                "required": ["effects"],
            },      
        }

        # razor_tool = {
        #     "name": "razor",
        #     "description": """
        #     Razor the timeline at a specific time
            
        #     """,
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "start": {
        #                 "type": "number",
        #                 "description": "The start time to razor the timeline",
        #             },
        #             "end": {
        #                 "type": "number",
        #                 "description": "The end time to razor the timeline",
        #             },
        #         },
        #         "required": ["start", "end"],
        #     },
        # }
            
            
        # add_text_tool = {
        #     "name": "add_text",
        #     "description": "Add a text to the timeline",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "text": {
        #                 "type": "string",
        #                 "description": "The text to add to the timeline",
        #             },
        #         },
        #         "required": ["text"],
        #     },
        # }
            


        tool_list = [move_item_tool, insert_item_tool, delete_item_tool, add_marker_tool, edit_effect_tool]

        
        
        
        
        full_prompt = f"""    
        ### Project Context :
        {json.dumps(project_final_context, indent=2)}

        ### Timeline Context :
        {json.dumps(final_context, indent=2)}
        
        ### User prompt : 
        {prompt}
            """.strip()

        system_instruction="""
            You are a professional video editor. You receive a JSON representation of a timeline and the project context. 
            You perform the user prompt, 
            modify the timeline,
            and return a list of actions to perform
            
            ### Rules: 
            - be careful on the track number to not overlap clips, audio or video
            - add marker to add informations or suggestions if needed
            - you must perform the actions in the right order
        """
        liste_actions = gemini_call(full_prompt, system_instruction, None, tool_list, 0.2, "gemini-2.5-flash-lite" )
        print("🎬 Actions timeline à exécuter:", liste_actions)
        
        # 4. Application des opérations à effectuer
        config.API_STATUS = "Performing actions..."
        for action in liste_actions:
            if action["name"] == "edit_effect":
                for effect in action["args"]["effects"]:
                    ID = effect["ID"]
                    
                    # retrouver le no de track associé à l'ID dans timeline_V0
                    for item in timeline_V0:
                        if item["ID"] == ID:
                            piste = item["piste"]
                            break
                    
                    prompt = effect["prompt"]
                    await apply_effect(ID, prompt, piste)
                liste_actions.remove(action)
                
            elif action["name"] == "delete_item":
                ID = action["args"]["ID"]
                for item in timeline_V0:
                    if item["ID"] == ID:
                        piste_type = item["piste"].split(" ")[0]
                        piste_number = int(item["piste"].split(" ")[1])
                        break
                action["args"]["piste_type"] = piste_type
                action["args"]["piste_number"] = piste_number

                liste_actions.remove(action)
                
            # ajouter le nodeId
            elif action["name"] == "move_item":
                ID = action["args"]["ID"]
                for item in timeline_V0:
                    if item["ID"] == ID:
                        nodeId = item["nodeId"]
                        piste_type = item["piste"].split(" ")[0]
                        piste_number = int(item["piste"].split(" ")[1])
                        break
                action["args"]["nodeId"] = nodeId
                action["args"]["piste_type"] = piste_type
                action["args"]["piste_number"] = piste_number

        
        
        call_id = str(uuid.uuid4())
        
        script_arg = json.dumps(liste_actions)
        script = f"$._MYFUNCTIONS.executeTimelineActionList({script_arg});"

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
        
        
        result = f"OBSERVATION : Timeline has been edited successfully for the prompt: ## PROMPT ## {prompt} . The following actions have been performed: {liste_actions}"
        
        
        return result
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            log_lignes = []
            for trace in tb:
                log_lignes.append(f"Fichier \"{trace.filename}\", ligne {trace.lineno}, dans {trace.name}")
            log_complet = "\n".join(log_lignes)
            message = f"Erreur edit_timeline_structure : {str(e)}\nTraceback complet :\n{log_complet}"
        else:
            message = f"Erreur edit_timeline_structure: {str(e)}"
        print(message)
        return "Error: " + message



# ------------- AGENT -------------



class GetCreativeTodo(BaseModel):
    prompt: str = Field(description="the prompt to get the creative todo")

@tool("get_creative_todo", args_schema=GetCreativeTodo)
async def get_creative_todo(prompt: str):
    """
    Ask the creative todo to perform the user prompt
    """
    
    
    system_instruction = f"""
    You are a creative video editor.
    You receive a user prompt.
    Your task is to think how to handle the user prompt, and return a to do list to describe to an AI agent how to handle the user prompt.
    
    
    ### TOOLS access:
    - get_project_structure : give informations about available media to use
    - edit_project_structure : use to create sequences, bins, organize the project structure. Can handle multiple actions in one call.
    - get_timeline_structure : use to get the structure of a timeline
    - edit_timeline_structure : use to add clips, audio, effects, edit a timeline etc...
    - open_timeline : use to open a timeline before editing it or grepping it
    - labelize_audio : use to get music tempo, transcription of audio before using it 
    
    
    you can use the followings common rules about video editing : 
    
    ### RULES:
    - if needed specifiy the tool to use in your todo point
    - in an edit based on music, you must follow the music tempo (downbeats)
    - you should arrange the clips based on their duration to not let space between clips
    - use the diffrents tracks number to make a clear edit, and add marker if needed 
    - be careful on the size of the clips vs the size of the timeline

    
    """
    
    structured_output = {
        "type": "array",
        "items": {
            "type": "string",
            "description": "The todo to perform the user prompt"
        }
    }
    
    todo_list = gemini_call(prompt, system_instruction, structured_output, None, 0.2, "gemini-2.5-flash" )
    
    print(f"Todo list")
    for todo in todo_list:
        print(f"- {todo}")
    
    return todo_list





class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
 
def create_agent_graph(model_name: str = "fast"):

    tools = [get_project_structure, edit_project_structure, 
             get_timeline_structure, edit_timeline_structure, 
             open_timeline,
            #  labelize_audio,
            #  get_creative_todo
             ]
    
    tool_node = ToolNode(tools)

    
    model = PremiereGPT_LLM(model=model_name, temperature=0.5)
    model = model.bind_tools(tools)
    
    def agent(state: AgentState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> str:
        if not state["messages"][-1].tool_calls:
            return "end"
        return "continue"
    
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()

async def run_agent_streaming(user_input: str,  existing_history: List[Dict[str, Any]] = None, token: str = None, model: str = "fast"):
    """Exécute l'agent et streame les résultats sous forme de JSON."""
    global AGENT_TOKEN
    # Définir le token global pour cette session
    AGENT_TOKEN = token
    app = create_agent_graph(model)

    # --- Gestion de l'historique ---
    conversation_history = [
        SystemMessage(content="""
    You are a an expert in Video Editing.
    You can use the tools to perform actions in Adobe Premiere Pro, 

    ### IMPORTANT:
    - Prefer using tools over guessing.
    - The tool get_creative_todo is usefull to think how to handle the user prompt 
    - Calling the tools get_project_structure and get_timeline_structure can be usefull if the user prompt is not clear.
    - If the result of the tool is not clear, you can ask the user to provide more details.       
    """),
    ]
    
    if existing_history:
        for message_data in existing_history:
            role = message_data.get("role")
            content = message_data.get("content")
            if role == "user":
                conversation_history.append(HumanMessage(content=content))
            elif role == "assistant":
                # Le contenu de l'assistant peut être un JSON stringified
                try:
                    # Essayons de parser le contenu si c'est un JSON
                    assistant_content_data = json.loads(content)
                    # Reconstruire un message simple pour l'historique, le LLM comprendra le contexte
                    formatted_content = assistant_content_data.get('content', '')
                    if assistant_content_data.get('tool_calls'):
                         formatted_content += f"\n(Action: Called tools {', '.join([t.get('name', 'unknown') for t in assistant_content_data['tool_calls']])})"
                    print(f"Formatted content: {formatted_content}")
                    conversation_history.append(AIMessage(content=formatted_content))
                except (json.JSONDecodeError, TypeError):
                    # Si ce n'est pas un JSON, on le prend tel quel
                    conversation_history.append(AIMessage(content=content))

    conversation_history.append(HumanMessage(content=user_input))
    # --------------------------------


    # for msg in conversation_history:
        # Affichage simplifié pour éviter de surcharger les logs
        # content_preview = str(msg.content)[:150] + '...' if len(str(msg.content)) > 150 else str(msg.content)
        # print(f"  - [{msg.type.upper()}]: {content_preview}")


    inputs = {"messages": conversation_history}
    
    thought_accumulator = ""
    tool_call_depth = 0  # profondeur d'appel des tools; 0 = appel top-level

    async for chunk in app.astream_events(inputs, version="v1"):
        kind = chunk["event"]
        
        if kind == "on_chat_model_stream":
            content = chunk["data"]["chunk"].content
            if content:
                # Conserver exactement le contenu streamé (y compris \n et espaces d'indentation)
                thought_accumulator += content
                yield {"type": "thought", "content": content}
                await asyncio.sleep(0.01)
        
        elif kind == "on_chat_model_end":
            if thought_accumulator:
                # This AIMessage is the decision to call a tool, let's log it as a thought.
                if 'tool_calls' in str(chunk['data']['output']): # Simple check
                    print(f"\n--- [THOUGHT] Agent decides to call a tool ---")
                else:
                    print(f"\n--- [THOUGHT] Agent's thought process ---")

                print("------------------------------------------")
                thought_accumulator = ""

        elif kind == "on_tool_start":
            tool_name = chunk["name"]
            tool_args = chunk["data"].get("input")
            print(f"\n--- [TOOL CALL] ---" + f"  - Name: `{tool_name}`" + f"  - Arguments: {tool_args}")

            # Afficher uniquement pour les appels top-level (pas ceux déclenchés par un autre tool)
            if tool_call_depth == 0:
                tool_info = TOOL_DISPLAY_MAPPING.get(tool_name, {"title": tool_name, "category": "default"})
                yield {
                    "type": "tool_start",
                    "title": tool_info["title"],
                    "category": tool_info["category"],
                    "args": tool_args
                }

            tool_call_depth += 1

        elif kind == "on_tool_end":
            tool_name = chunk["name"]
            tool_output = chunk["data"].get("output")
            
            output_preview = (str(tool_output)[:300] + '...') if len(str(tool_output)) > 300 else str(tool_output)
            output_preview = output_preview.replace('\n', ' ').replace('\r', ' ').replace('    ', ' ').replace('  ', ' ')
            print(f"\n--- [TOOL RESULT] --- - From: `{tool_name}` - Output: {output_preview}")

            # Diminuer la profondeur et n'émettre l'événement de fin que pour le top-level
            tool_call_depth = max(0, tool_call_depth - 1)
            if tool_call_depth == 0:
                tool_info = TOOL_DISPLAY_MAPPING.get(tool_name, {"title": tool_name, "category": "default"})
                yield {
                    "type": "tool_end",
                    "title": tool_info["title"]
                }

        elif kind == "on_chain_end":
            if chunk["name"] == "LangGraph":
                output = chunk.get("data", {}).get("output")
                if output and isinstance(output, dict) and "messages" in output:
                    messages = output["messages"]
                    if messages and isinstance(messages, list):
                        final_answer = messages[-1].content
                        print(f"\n--- [FINAL ANSWER] ---")
                        print(final_answer)
                        print("----------------------")
                        config.API_STATUS = "End"
                        yield {"type": "answer", "content": final_answer}