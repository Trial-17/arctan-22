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
import platform
from enum import Enum
from pydantic import BaseModel, Field
from pathlib import Path

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage, HumanMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import av

from LIB import config
from LIB.config import PENDING_JS_CALLS, MODEL_REACT_PROJECT_STRUCTURE, MODEL_REACT_PROJECT_STRUCTURE_TOOL
from LIB.music_analysis_V2 import analyze_music
from LIB.subtitles import main_transcription_for_agent
from LIB.custom_llm import PremiereGPT_LLM

from LIB.AGENT_project import get_project_structure
from LIB.config import EDIT_TIMELINE_STRUCTURE_TOOL_LIST


def get_mogrt_path():
    """
    Resolve the MOGRT file path dynamically based on the current environment.
    Similar to preset path resolution pattern.
    
    Returns:
        str: Absolute path to the MOGRT file
        
    Raises:
        FileNotFoundError: If the MOGRT file cannot be found
    """
    # Try to find MOGRT in the CEP extension directory
    project_root = Path(__file__).parent.parent.parent.parent
    mogrt_path = project_root / "mogrt" / "Texte PremiereGPT.mogrt"

    # Validate that the file exists
    if not mogrt_path.exists():
        mogrt_path = "/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/mogrt/Texte PremiereGPT.mogrt"
    
    return str(mogrt_path)

def gptoss_call(prompt, system_instruction, structured_output = None, tool_list = None, temperature= 0.2,  model = "openai/gpt-oss-120b"): 


    try : 
        payload = {
            "prompt": str(prompt),
            "system_instruction": str(system_instruction),
            "temperature": temperature,
            "model": model, 
            "structured_output": structured_output
        }


        response = requests.post(
            f"{config.API_URL}/gpt-call",
            json=payload,
            headers={"Authorization": f"Bearer {config.AGENT_TOKEN}"}
        )

        # Vérifier le statut de la réponse
        if response.status_code != 200:
            raise Exception(f"Erreur API (status {response.status_code}): {response.text}")
        
        # Extraire le résultat
        result = response.json().get("result")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors de l'appel à l'API relai: {str(e)}")
        raise Exception(f"Erreur lors de l'appel à l'API relai: {str(e)}")



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
        

        
        # Retourner le résultat dans le même format que l'ancienne fonction
        return result
        
    except requests.exceptions.RequestException as e:

        print(f"❌ Erreur lors de l'appel à l'API relai: {str(e)}")
        raise Exception(f"Erreur lors de l'appel à l'API relai: {str(e)}")


# ------------- TIMELINE MANAGEMENT -------------
# donner la taille et orientation de la timeline

async def add_text(texts):
    """
    Add one or more texts to the timeline
    """
    try:
        full_prompt = f"""    
        ### AGENT PROMPT : 
        {texts}
        """.strip()
        
        project_effect_instruction=f"""
            You are a professional video editor expert in Premiere Pro.
            Your task is to add texts to the timeline based on the user prompt.
            
            ### RULES : 
            - follow the user instructions
            - follow the format of the available controls
            - only return the controls that are needed to perform the user instructions
            - think about the size and position of the text to display
            
            
            ### AVAILABLE CONTROLS & FORMAT : 
            #### TEXT : 
            - text: string
            - font_size: float
            - text_color: [a, r, g, b]
            
            #### ANIMATION : 
            - enable_animation: true or false
            - animation_duration: float (in seconds)
            - animation_based: integer, 0 for Characters, 1 for CharacterExcludingSpaces, 2 for Words, 3 for Lines
            - animation_slide_position: tuple [x, y]
            
            #### SHADOW : 
            - shadow_color: [a, r, g, b]
            - shadow_opacity: integer (0-100)
            - shadow_angle: integer (0-360)
            - shadow_distance: integer
            - shadow_spread: integer 
            - shadow_size: integer 
            
            #### BOX : 
            - box_color: [a, r, g, b]
            - box_opacity: integer (0-100)
            - box_roundness: integer (0-100)
            - box_padding_vertical: float (pixels, 0-500)
            - box_padding_horizontal: float (pixels, 0-500)
            - box_stroke_color: [a, r, g, b]
            - box_stroke_width: float
            
            #### BACKGROUND : 
            - background_color: [a, r, g, b]
            - background_opacity: integer (0-100)
        """.strip()
        

        add_text_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # TEXT
                    "text": {
                        "type": "string",
                        "description": "The text content to display"
                    },
                    "font_size": {
                        "type": "number",
                        "description": "The font size"
                    },
                    "text_color": {
                        "type": "array",
                        "description": "Text color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 4,
                        "maxItems": 4
                    },
                    
                    # ANIMATION
                    "enable_animation": {
                        "type": "boolean",
                        "description": "Enable or disable animation"
                    },
                    "animation_duration": {
                        "type": "number",
                        "description": "Animation duration in seconds"
                    },
                    "animation_based": {
                        "type": "integer",
                        "description": "Animation type: 0 for Characters, 1 for CharacterExcludingSpaces, 2 for Words, 3 for Lines"
                    },
                    "animation_slide_position": {
                        "type": "array",
                        "description": "Animation slide position [x, y]",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 2,
                        "maxItems": 2
                    },
                    
                    # SHADOW
                    "shadow_color": {
                        "type": "array",
                        "description": "Shadow color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "shadow_opacity": {
                        "type": "integer",
                        "description": "Shadow opacity (0-100)",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "shadow_angle": {
                        "type": "integer",
                        "description": "Shadow angle in degrees (0-360)",
                        "minimum": 0,
                        "maximum": 360
                    },
                    "shadow_distance": {
                        "type": "integer",
                        "description": "Shadow distance"
                    },
                    "shadow_spread": {
                        "type": "integer",
                        "description": "Shadow spread"
                    },
                    "shadow_size": {
                        "type": "integer",
                        "description": "Shadow size"
                    },
                    
                    # BOX
                    "box_color": {
                        "type": "array",
                        "description": "Box background color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "box_opacity": {
                        "type": "integer",
                        "description": "Box opacity (0-100)",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "box_roundness": {
                        "type": "integer",
                        "description": "Box corner roundness (0-100)",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "box_padding_vertical": {
                        "type": "number",
                        "description": "Box vertical padding in pixels (0-500)",
                        "minimum": 0,
                        "maximum": 500
                    },
                    "box_padding_horizontal": {
                        "type": "number",
                        "description": "Box horizontal padding in pixels (0-500)",
                        "minimum": 0,
                        "maximum": 500
                    },
                    "box_stroke_color": {
                        "type": "array",
                        "description": "Box stroke color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "box_stroke_width": {
                        "type": "number",
                        "description": "Box stroke width"
                    },
                    
                    # BACKGROUND
                    "background_color": {
                        "type": "array",
                        "description": "Background color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "background_opacity": {
                        "type": "integer",
                        "description": "Background opacity (0-100)",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["text"]
            }
        }
        
        add_text_response = gemini_call(full_prompt, project_effect_instruction, add_text_schema, None, 0.2, "gemini-2.5-flash-lite")
        
        results = []
        
        # Get MOGRT path once for all text imports
        try:
            mogrt_path = get_mogrt_path()
            # Escape backslashes for Windows paths and forward slashes for proper JSON escaping
            mogrt_path_escaped = mogrt_path.replace("\\", "\\\\")
        except (FileNotFoundError, OSError) as e:
            return f"Error: {str(e)}"
        


        for i in range(min(len(add_text_response), len(texts))):
            text_config = add_text_response[i]
            text = texts[i]
            # 1. Importer le mogrt (créer le texte)
            call_id = str(uuid.uuid4())
            import_script = f'$._MYFUNCTIONS.importMogrtAtTime({text["start"]}, {text["duration"]}, "{mogrt_path_escaped}");'


            PENDING_JS_CALLS[call_id] = {
                "args": {"script": import_script},
                "result": None,
                "status": "pending"
            }
            
            timeout = 30
            start_time = time.time()
            import_result = None
            while time.time() - start_time < timeout:
                if call_id in PENDING_JS_CALLS and PENDING_JS_CALLS[call_id]["status"] == "completed":
                    import_result = PENDING_JS_CALLS[call_id]["result"]
                    del PENDING_JS_CALLS[call_id]
                    break
                await asyncio.sleep(0.5)
            import_result = json.loads(import_result)
            
            
            if call_id in PENDING_JS_CALLS:
                del PENDING_JS_CALLS[call_id]
            
            if import_result['success'] == False:
                return "Error: " + import_result['message']




            # 2. Éditer le texte avec les propriétés
            edit_args = {
                "ID": import_result['nodeId'],
                "no_piste": import_result['no_piste'],
                "settings": text_config
            }
            
            call_id = str(uuid.uuid4())
            edit_script = f"$._MYFUNCTIONS.editText({json.dumps(edit_args)});"
            
            PENDING_JS_CALLS[call_id] = {
                "args": {"script": edit_script},
                "result": None,
                "status": "pending"
            }
            
            # Attendre la réponse
            start_time = time.time()
            edit_result = None
            while time.time() - start_time < timeout:
                if call_id in PENDING_JS_CALLS and PENDING_JS_CALLS[call_id]["status"] == "completed":
                    edit_result = PENDING_JS_CALLS[call_id]["result"]
                    edit_result = json.loads(edit_result)
                    del PENDING_JS_CALLS[call_id]
                    break
                await asyncio.sleep(0.5)
            
            if call_id in PENDING_JS_CALLS:
                del PENDING_JS_CALLS[call_id]
            
        
        return json.dumps({"status": "completed", "results": edit_result})


    except Exception as e:
        print(f"Erreur add_text: {str(e)}")
        return "Error: " + str(e)

async def modify_text(ID, no_piste, prompt): 
    """
    Modifiy one text on the timeline
    """
    try:
        full_prompt = f"""    
        ### AGENT PROMPT : 
        {prompt}
        """.strip()
        
        project_effect_instruction=f"""
            You are a professional video editor expert in Premiere Pro.
            Your task is to modify one text on the timeline based on the user prompt.
            
            ### RULES : 
            - follow the user instructions
            - follow the format of the available controls
            - only return the controls that are needed to perform the user instructions
            - think about the size and position of the text to display
            
            
            ### AVAILABLE CONTROLS & FORMAT : 
            #### TEXT : 
            - text: string
            - font_size: float
            - text_color: [a, r, g, b]
            
            #### ANIMATION : 
            - enable_animation: true or false
            - animation_duration: float (in seconds)
            - animation_based: integer, 0 for Characters, 1 for CharacterExcludingSpaces, 2 for Words, 3 for Lines
            - animation_slide_position: tuple [x, y]
            
            #### SHADOW : 
            - shadow_color: [a, r, g, b]
            - shadow_opacity: integer (0-100)
            - shadow_angle: integer (0-360)
            - shadow_distance: integer
            - shadow_spread: integer 
            - shadow_size: integer 
            
            #### BOX : 
            - box_color: [a, r, g, b]
            - box_opacity: integer (0-100)
            - box_roundness: integer (0-100)
            - box_padding_vertical: float (pixels, 0-500)
            - box_padding_horizontal: float (pixels, 0-500)
            - box_stroke_color: [a, r, g, b]
            - box_stroke_width: float
            
            #### BACKGROUND : 
            - background_color: [a, r, g, b]
            - background_opacity: integer (0-100)
        """.strip()
        

        modify_text_schema = {
            "type": "object",
            "properties": {
                # TEXT
                "text": {
                    "type": "string",
                    "description": "The text content to display"
                },
                "font_size": {
                    "type": "number",
                    "description": "The font size"
                },
                "text_color": {
                    "type": "array",
                    "description": "Text color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 4,
                    "maxItems": 4
                },
                
                # ANIMATION
                "enable_animation": {
                    "type": "boolean",
                    "description": "Enable or disable animation"
                },
                "animation_duration": {
                    "type": "number",
                    "description": "Animation duration in seconds"
                },
                "animation_based": {
                    "type": "integer",
                    "description": "Animation type: 0 for Characters, 1 for CharacterExcludingSpaces, 2 for Words, 3 for Lines"
                },
                "animation_slide_position": {
                    "type": "array",
                    "description": "Animation slide position [x, y]",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                
                # SHADOW
                "shadow_color": {
                    "type": "array",
                    "description": "Shadow color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 4,
                    "maxItems": 4
                },
                "shadow_opacity": {
                    "type": "integer",
                    "description": "Shadow opacity (0-100)",
                    "minimum": 0,
                    "maximum": 100
                },
                "shadow_angle": {
                    "type": "integer",
                    "description": "Shadow angle in degrees (0-360)",
                    "minimum": 0,
                    "maximum": 360
                },
                "shadow_distance": {
                    "type": "integer",
                    "description": "Shadow distance"
                },
                "shadow_spread": {
                    "type": "integer",
                    "description": "Shadow spread"
                },
                "shadow_size": {
                    "type": "integer",
                    "description": "Shadow size"
                },
                
                # BOX
                "box_color": {
                    "type": "array",
                    "description": "Box background color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 4,
                    "maxItems": 4
                },
                "box_opacity": {
                    "type": "integer",
                    "description": "Box opacity (0-100)",
                    "minimum": 0,
                    "maximum": 100
                },
                "box_roundness": {
                    "type": "integer",
                    "description": "Box corner roundness (0-100)",
                    "minimum": 0,
                    "maximum": 100
                },
                "box_padding_vertical": {
                    "type": "number",
                    "description": "Box vertical padding in pixels (0-500)",
                    "minimum": 0,
                    "maximum": 500
                },
                "box_padding_horizontal": {
                    "type": "number",
                    "description": "Box horizontal padding in pixels (0-500)",
                    "minimum": 0,
                    "maximum": 500
                },
                "box_stroke_color": {
                    "type": "array",
                    "description": "Box stroke color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 4,
                    "maxItems": 4
                },
                "box_stroke_width": {
                    "type": "number",
                    "description": "Box stroke width"
                },
                
                # BACKGROUND
                "background_color": {
                    "type": "array",
                    "description": "Background color in [a, r, g, b] format where a is alpha/opacity (0-1) and r,g,b are color values (0-255)",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 4,
                    "maxItems": 4
                },
                "background_opacity": {
                    "type": "integer",
                    "description": "Background opacity (0-100)",
                    "minimum": 0,
                    "maximum": 100
                }
            }
        }
        
        modify_text_response = gemini_call(full_prompt, project_effect_instruction, modify_text_schema, None, 0.2, "gemini-2.5-flash-lite")
        
        results = []

        # 2. Éditer le texte avec les propriétés
        edit_args = {
            "ID": ID,
            "no_piste": int(no_piste.split(" ")[1]),
            "settings": modify_text_response
        }
        
        call_id = str(uuid.uuid4())
        edit_script = f"$._MYFUNCTIONS.editText({json.dumps(edit_args)});"
        
        # print(edit_script)
        
        PENDING_JS_CALLS[call_id] = {
            "args": {"script": edit_script},
            "result": None,
            "status": "pending"
        }
        
        # Attendre la réponse
        timeout = 30
        start_time = time.time()
        edit_result = None
        while time.time() - start_time < timeout:
            if call_id in PENDING_JS_CALLS and PENDING_JS_CALLS[call_id]["status"] == "completed":
                edit_result = PENDING_JS_CALLS[call_id]["result"]
                edit_result = json.loads(edit_result)
                del PENDING_JS_CALLS[call_id]
                break
            await asyncio.sleep(0.5)
        
        if call_id in PENDING_JS_CALLS:
            del PENDING_JS_CALLS[call_id]
        
        # print(edit_result)
        
        return json.dumps({"status": "completed", "results": edit_result})


    except Exception as e:
        print(f"Erreur modify_text: {str(e)}")
        return "Error: " + str(e)



async def apply_effect(ID, prompt, piste, seq_width, seq_height, item_width, item_height):
    """
    Détermine :
    - le nom de l'effet
    - si on doit le créer ou modifier
    - le nom des propriétés à modifier, et leurs valeurs à modifier
    - les éventuelles keyframes
    """
    
    #1. Récupérer le nom de l'effet, ses propriétés

    lumetri_mapping = {
        "Temperature": 14,
        "Tint": 15,
        "Saturation": 16,
        "Exposure": 19,
        "Contrast": 20,
        "Highlights": 21,
        "Shadows": 22,
        "Whites": 23,
        "Blacks": 24
    }

    motion_mapping = {
        "Position": 0,
        "Scale": 1,
        "Rotation": 4,
        "AnchorPoint": 5,
        "AntiFlickerFilter": 6, 
        "CropLeft": 7,
        "CropTop": 8,
        "CropRight": 9,
        "CropBottom": 10,
    }

    effects = [

        {
            "name": "AE.ADBE Easy Motion",
            "description": """
            to edit easily position and crop
            define the grid and the position in the grid of your element. 

            Letter are for the vertical axis
            Number are for the horizontal axis
            A1 is bottom left


            Exemple 1: 
            - grid in 2x2 (square grid of 2 rows and 2 columns)
            - element in A1 to A2
            -> the element will occupy all the bottom space

            Exemple 2: 
            - grid in 5x1 (so it's a vertical grid of 5 rows with 1 column)
            - element in C1
            -> the element will occupy the middle of the frame, on the whole width, and 1/5 of the height

            Exemple 3: 
            - grid in 1x1 (whole frame)
            - element in A1
            -> the element will occupy the whole frame


            """,
            1: "grid vertical tile number", 
            2: "grid horizontal tile number", 
            3: "element start tile coordinate", 
            4: "element end tile coordinate", 
            5: "Boolean, True to fit (resize) the element, False to crop"
        }, 


    ]
    
    # 2. Appliquer les effets
    type_piste = piste.split(" ")[0]
    no_piste = int(piste.split(" ")[1])
    
    try: 

        # 1.1 Choix des noms des effets

        full_prompt = f"""    
        ### USER PROMPT : 
        {prompt}
        """.strip()
            
        project_effect_instruction=f"""
        You are a professional visual effects supervisor in Premiere Pro. 
        Your task is to select the right effects to perform the user prompt.

        You can add many effect or modify many properties of the same effect in the same call.
        If you just want to add effect without any property, simply call the effect name.

        Only call parameters that you need to modify. Don't call parameters that you don't need to modify.

        Prefere using the "AE.ADBE Easy Motion" effect to edit easily position and crop, for exemple if you want to set the item on a specific position in the frame.

        If you use a motion tool, you must use the knowledge for motion below to edit the motion in the good proportion.
        ### KNOWLEDGE FOR MOTION: 

        - width of the sequence: {seq_width}
        - height of the sequence: {seq_height}
        - width of the item: {item_width}
        - height of the item: {item_height}
        
        """
        

        effects_properties = gemini_call(full_prompt, project_effect_instruction, None, config.EFFECT_TOOL_LIST, 0.5, config.MODEL_EFFECT )

        if not effects_properties or effects_properties == []:
            return "This effect is not available in Premiere GPT for now"


        print(effects_properties)


        processed_properties = []


        
        for effect_to_add in effects_properties:
            
            if effect_to_add['name'] == 'Opacity': 

                effect_property = {
                    "effect_name": "AE.ADBE Opacity",
                    "property_number": 0,
                    "property_value": effect_to_add['args']['Opacity'],
                }
                processed_properties.append(effect_property)

            elif effect_to_add['name'] == "Lumetri":
                for property_ in effect_to_add['args']:
                    effect_property = {
                        "effect_name": "AE.ADBE Lumetri",
                        "property_number": lumetri_mapping[property_],
                        "property_value": effect_to_add['args'][property_],
                    }
                    processed_properties.append(effect_property)

            elif effect_to_add['name'] == "Motion":
                for property_ in effect_to_add['args']:
                    effect_property = {
                        "effect_name": "AE.ADBE Motion",
                        "property_number": motion_mapping[property_],
                        "property_value": effect_to_add['args'][property_],
                    }
                    processed_properties.append(effect_property)


            elif effect_to_add['name'] == "EasyMotion":


                print("Sequence width: ", seq_width, "Sequence height: ", seq_height, "Item width: ", item_width, "Item height: ", item_height)

                unity_vertical = seq_height / effect_to_add['args']['GridVerticalTiles']
                unity_horizontal = seq_width / effect_to_add['args']['GridHorizontalTiles']
                end_tiles = effect_to_add['args']['ElementEndTile']
                start_tiles = effect_to_add['args']['ElementStartTile']



                print("Unity vertical: ", unity_vertical, "Unity horizontal: ", unity_horizontal)
                print("Element start tile: ", start_tiles, "Element end tile: ", end_tiles)


                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


                #0. Définir la taille finale de l'élément
                target_pixel_width = (int(end_tiles[1]) - int(start_tiles[1]) + 1) * unity_horizontal
                target_pixel_height = (alphabet.index(end_tiles[0]) - alphabet.index(start_tiles[0]) +1) * unity_vertical

                print("Target pixel width: ", target_pixel_width, "Target pixel height: ", target_pixel_height)

                ratio = max(target_pixel_width / item_width, target_pixel_height / item_height)

                print("Ratio: ", ratio)

                processed_properties.append({
                    "effect_name": "AE.ADBE Motion",
                    "property_number": motion_mapping["Scale"],
                    "property_value": ratio*100,
                })


                new_width = item_width * ratio
                new_height = item_height * ratio
                print("New width: ", new_width, "New height: ", new_height)


                if new_width != target_pixel_width:
                    nb_pixel_to_delete = new_width - target_pixel_width
                    crop_ratio = (nb_pixel_to_delete / item_width) / 2 / ratio *100
                    processed_properties.append({
                        "effect_name": "AE.ADBE Motion",
                        "property_number": motion_mapping["CropLeft"],
                        "property_value": crop_ratio,
                    })
                    processed_properties.append({
                        "effect_name": "AE.ADBE Motion",
                        "property_number": motion_mapping["CropRight"],
                        "property_value": crop_ratio,
                    })
                    print("Crop left: ", crop_ratio, "Crop right: ", crop_ratio)

                if new_height != target_pixel_height:
                    nb_pixel_to_delete = new_height - target_pixel_height
                    crop_ratio = (nb_pixel_to_delete / item_height) / 2 / ratio *100
                    processed_properties.append({
                        "effect_name": "AE.ADBE Motion",
                        "property_number": motion_mapping["CropTop"],
                        "property_value": crop_ratio,
                    })
                    processed_properties.append({
                        "effect_name": "AE.ADBE Motion",
                        "property_number": motion_mapping["CropBottom"],
                        "property_value": crop_ratio,
                    })
                    print("Crop top: ", crop_ratio, "Crop bottom: ", crop_ratio)


                #1. Définir la position finale de l'élément


                start_Y = alphabet.index(end_tiles[0]) * unity_vertical
                start_X = int(end_tiles[1]) * unity_horizontal
                end_Y = start_Y + target_pixel_height
                end_X = start_X + target_pixel_width

                print("Start X: ", start_X, "Start Y: ", start_Y)

                print("End X: ", end_X, "End Y: ", end_Y)

                position_Y = (end_Y - start_Y) / 2 + start_Y
                position_X = (end_X - start_X) / 2 + start_X

                print("Position X: ", position_X, "Position Y: ", position_Y)

                actual_X = seq_width/2
                actual_Y = seq_height/2

                final_X = ((position_X - actual_X) / seq_width + 0.5) 
                final_Y = (-(position_Y - actual_Y) / seq_height + 0.5)

                print("Final X: ", final_X, "Final Y: ", final_Y)

                processed_properties.append({
                    "effect_name": "AE.ADBE Motion",
                    "property_number": motion_mapping["Position"],
                    "property_value": [final_X, final_Y],
                })

                print("List adjustments: ", processed_properties)






        # 3. Appliquer les effets traités
        for effect_property in processed_properties:

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
            

        return "All this effects have been applied successfully" + str(effects_properties)
    
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            log_lignes = []
            for trace in tb:
                log_lignes.append(f"Fichier \"{trace.filename}\", ligne {trace.lineno}, dans {trace.name}")
            log_complet = "\n".join(log_lignes)
            message = f"Erreur adding effect : {str(e)}\nTraceback complet :\n{log_complet}"
        else:
            message = f"Erreur adding effect: {str(e)}"
        print(message)
        return "Error: " + message




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
        - Return the sequenceId of the timeline
        
        ### PROJECT
        {project_V0}

        """
        
        
        structured_output = {
            "type": "object",
            "properties": {
                "sequenceId": {"type": "string", "description": "The sequenceId of the timeline to open. NOT the nodeId"}
            },
            "required": ["sequenceId"],
        }
        
        sequenceId = gemini_call(timeline_name, system_instruction, structured_output, None, 0.2, "gemini-2.5-flash-lite" )['sequenceId']
        
        
        # 2. Appel du script pour ouvrir la timeline
        call_id = str(uuid.uuid4())
        
        script = f"$._MYFUNCTIONS.activateSequenceById('{sequenceId}');"
        
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
    include_metadata :bool = Field(default=False, description="Use it only if you need to know the description of the video and images to perform the task. Slower")
    include_effects :bool = Field(default=False, description="Use it only if you need to edit effects. Slower")

@tool("get_timeline_structure", args_schema=GetTimelineStructure)
async def get_timeline_structure(include_metadata: bool = False, include_effects: bool = False):
    """
    Returns the JSON structure of the active timeline in Premiere Pro
    Usefull to read the timeline informations

    """

    # - USE open_timeline to open the timeline before getting the structure if needed. 
    # - USE Metadata only if you need to know the transcription of audio, the music tempo, the description of the clips or images to perform the task
    # - USE Effects only if you need to know the effects of the clips to perform the task
    try:
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


        result = json.loads(result)

        width = result[-5]["width"]
        height = result[-4]["height"]
        FPS = result[-3]["FPS"]
        inSelection = max(0, result[-2]["inSelection"])
        outSelection = 1e9 if result[-1]["outSelection"] <= 0 else result[-1]["outSelection"]
        result = result[:-5]
        # print(f"inSelection: {inSelection}, outSelection: {outSelection}")
        

        
        # 0. Garder les items dans la selection 
        selected_items = []
        for clip in result:
            if inSelection <= clip["start"] <= outSelection:
                selected_items.append(clip)
            elif inSelection <= clip["end"] <= outSelection:
                selected_items.append(clip)
            elif clip["start"] <= inSelection and clip["end"] >= outSelection:
                selected_items.append(clip)
        
        for clip in selected_items:
            clip["start"] = max(clip["start"], inSelection)
            clip["end"] = min(clip["end"], outSelection)
            
                
                
                
        result = selected_items.copy()
        
        
        # 1. Chargement des effets
        if not include_effects:
            for clip in result:
                clip.pop('effects')
                
        
        # 2. Chargement des métadonnées
        project_V0 = await get_project_structure.ainvoke({"include_metadata": include_metadata, "include_audio": True})
        project_V0 = json.loads(project_V0)
        

        for clip in result:
            if "nodeId" in clip and clip["nodeId"]:
                project_item = find_item_by_node_id(project_V0.get("children", []), clip["nodeId"])
                
                if project_item and "metadata" in project_item:
                    if include_metadata:
                        clip["metadata"] = project_item["metadata"]
                    elif "downbeats" in project_item["metadata"]:
                        clip["downbeats"] = project_item["metadata"]["downbeats"]
                        
                        # 0. Garder les downbeats utilisé
                        final_downbeats = [downbeat - clip["inPoint"] + clip["start"]  for downbeat in clip["downbeats"] if downbeat >= clip["inPoint"] and downbeat <= clip["outPoint"]]
                        
                        # 1. Mettre les downbeats en fonction des FPS
                        liste_TC = [x/FPS for x in range(int(FPS))]
                        final_downbeats_decimal = [downbeat % 1 for downbeat in final_downbeats]

                        final_downbeats_decimal_inf = [max(x for x in liste_TC if x <= downbeat_decimal) for downbeat_decimal in final_downbeats_decimal]
                        final_downbeats = [downbeat - downbeat_decimal + downbeat_decimal_inf for downbeat, downbeat_decimal, downbeat_decimal_inf in zip(final_downbeats, final_downbeats_decimal, final_downbeats_decimal_inf)]
                        
                        clip["downbeats"] = final_downbeats
                    
                    if "segments" in project_item["metadata"]: 
                        segments = project_item["metadata"]["segments"]
                        # print(clip)

                        segment_final = []
                        for segment in segments : 
                            
                            if segment["start"] >= clip["inPoint"] and segment["start"] <= clip["outPoint"]:
                                segment_final.append(segment)
                            elif segment["end"] >= clip["inPoint"] and segment["end"] <= clip["outPoint"]:
                                segment_final.append(segment)
                            elif segment["start"] <= clip["inPoint"] and segment["end"] >= clip["outPoint"]:
                                segment_final.append(segment)


                        for segment in segment_final:
                            segment["start"] -= clip["inPoint"]
                            segment["end"] -= clip["inPoint"]

                        clip["segments"] = segment_final.copy()

                        
          
        # 3. supprimer les item audio qui sont liés a une video et qui n'ont pas de metadata
        video_items = [item for item in result if 'video' in item.get('piste', '')]
        audio_items = [item for item in result if 'audio' in item.get('piste', '')]

        # Pour comparer les metadonnées (dictionnaires), on les convertit en chaînes JSON triées
        def get_metadata_str(item):
            return json.dumps(item.get('metadata', {}), sort_keys=True)


        video_signatures = {
            (item['nodeId'], item['start'], item['end'], get_metadata_str(item))
            for item in video_items
        }

        # Conserver uniquement les éléments audio qui n'ont pas de doublon vidéo correspondant
        audio_items_uniques = []
        for item in audio_items:
            audio_signature = (item['nodeId'], item['start'], item['end'], get_metadata_str(item))
            if audio_signature not in video_signatures or item.get("segments", []) != []:
                audio_items_uniques.append(item)
                
                

        # Le résultat final est la combinaison de tous les éléments vidéo et des audios uniques
        result = video_items + audio_items_uniques
    
        return result, FPS, width, height
    
    except Exception as e:
        print(f"Erreur get_timeline_structure: {str(e)}")
        return "Error: " + str(e)
    

def compare_timeline_structures(initial_structure: List[Dict], final_structure: List[Dict]) -> Dict:
    """
    Compare deux structures de timeline et retourne les différences.
    Détecte: clips ajoutés, clips supprimés, clips déplacés (temps ou piste), clips modifiés (trim/durée)
    
    Args:
        initial_structure: Liste des clips de la timeline initiale
        final_structure: Liste des clips de la timeline finale
        
    Returns:
        Dict contenant les différences détectées et une observation textuelle
    """
    differences = {
        "items_added": [],      # Nouveaux clips ajoutés
        "items_removed": [],    # Clips supprimés
        "items_moved_time": [], # Clips déplacés dans le temps
        "items_moved_track": [], # Clips changés de piste
        "items_trimmed": [],    # Clips dont la durée a changé (trim)
        
        "texts_added": [],      # Nouveaux textes ajoutés
        "texts_removed": [],    # Textes supprimés
        "texts_moved_time": [], # Textes déplacés dans le temps
        "texts_moved_track": [], # Textes changés de piste
        "texts_trimmed": [],    # Textes dont la durée a changé (trim)
        
        "effects_added": [],    # Effets ajoutés à des clips/textes
        "effects_modified": [], # Effets modifiés sur des clips/textes
    }
    
    try:
        # print("initial_structure: ", initial_structure)
        # print("final_structure: ", final_structure)
        # Convertir en dict si nécessaire
        if isinstance(initial_structure, str):
            initial_structure = json.loads(initial_structure)
        if isinstance(final_structure, str):
            final_structure = json.loads(final_structure)
        
        # Créer des index par ID pour faciliter la comparaison
        initial_index = {item["ID"]: item for item in initial_structure}
        final_index = {item["ID"]: item for item in final_structure}
        
        # Détecter les clips/textes ajoutés
        for item_id, item_data in final_index.items():
            if item_id not in initial_index:
                item_info = {
                    "ID": item_id,
                    "nodeId": item_data.get("nodeId"),
                    "start": item_data.get("start"),
                    "end": item_data.get("end"),
                    "duration": round(item_data.get("end", 0) - item_data.get("start", 0), 2),
                    "piste": item_data.get("piste")
                }
                # Séparer les textes des clips
                if item_data.get("isText", False):
                    differences["texts_added"].append(item_info)
                else:
                    differences["items_added"].append(item_info)
        
        # Détecter les clips/textes supprimés
        for item_id, item_data in initial_index.items():
            if item_id not in final_index:
                item_info = {
                    "ID": item_id,
                    "nodeId": item_data.get("nodeId"),
                    "start": item_data.get("start"),
                    "end": item_data.get("end"),
                    "duration": round(item_data.get("end", 0) - item_data.get("start", 0), 2),
                    "piste": item_data.get("piste")
                }
                # Séparer les textes des clips
                if item_data.get("isText", False):
                    differences["texts_removed"].append(item_info)
                else:
                    differences["items_removed"].append(item_info)
        
        # Détecter les modifications sur les clips existants
        for item_id in initial_index:
            if item_id in final_index:
                initial_item = initial_index[item_id]
                final_item = final_index[item_id]
                
                initial_start = initial_item.get("start", 0)
                initial_end = initial_item.get("end", 0)
                final_start = final_item.get("start", 0)
                final_end = final_item.get("end", 0)
                
                initial_piste = initial_item.get("piste", "")
                final_piste = final_item.get("piste", "")
                
                # Vérifier si changement de piste
                if initial_piste != final_piste:
                    track_info = {
                        "ID": item_id,
                        "nodeId": final_item.get("nodeId"),
                        "old_piste": initial_piste,
                        "new_piste": final_piste,
                        "start": final_start,
                        "end": final_end
                    }
                    # Séparer les textes des clips
                    if final_item.get("isText", False):
                        differences["texts_moved_track"].append(track_info)
                    else:
                        differences["items_moved_track"].append(track_info)
                
                # Vérifier si déplacement dans le temps (sans changement de durée)
                initial_duration = initial_end - initial_start
                final_duration = final_end - final_start
                
                if (abs(initial_start - final_start) > 0.01 or abs(initial_end - final_end) > 0.01):
                    # Si la durée a changé, c'est un trim
                    if abs(initial_duration - final_duration) > 0.01:
                        trim_info = {
                            "ID": item_id,
                            "nodeId": final_item.get("nodeId"),
                            "piste": final_piste,
                            "old_start": round(initial_start, 2),
                            "old_end": round(initial_end, 2),
                            "old_duration": round(initial_duration, 2),
                            "new_start": round(final_start, 2),
                            "new_end": round(final_end, 2),
                            "new_duration": round(final_duration, 2),
                            "duration_change": round(final_duration - initial_duration, 2)
                        }
                        # Séparer les textes des clips
                        if final_item.get("isText", False):
                            differences["texts_trimmed"].append(trim_info)
                        else:
                            differences["items_trimmed"].append(trim_info)
                    # Sinon c'est juste un déplacement
                    else:
                        move_info = {
                            "ID": item_id,
                            "nodeId": final_item.get("nodeId"),
                            "piste": final_piste,
                            "old_start": round(initial_start, 2),
                            "new_start": round(final_start, 2),
                            "time_shift": round(final_start - initial_start, 2),
                            "duration": round(final_duration, 2)
                        }
                        # Séparer les textes des clips
                        if final_item.get("isText", False):
                            differences["texts_moved_time"].append(move_info)
                        else:
                            differences["items_moved_time"].append(move_info)
                
                # Comparer les effets
                initial_effects = initial_item.get("effects", [])
                final_effects = final_item.get("effects", [])
                
                # Créer un index des effets par matchName
                initial_effects_index = {effect.get("matchName"): effect for effect in initial_effects}
                final_effects_index = {effect.get("matchName"): effect for effect in final_effects}
                
                # Détecter les effets ajoutés
                for effect_name, effect_data in final_effects_index.items():
                    if effect_name not in initial_effects_index:
                        effect_info = {
                            "item_ID": item_id,
                            "nodeId": final_item.get("nodeId"),
                            "isText": final_item.get("isText", False),
                            "effect_name": effect_name,
                            "piste": final_piste,
                            "start": final_start,
                            "end": final_end
                        }
                        differences["effects_added"].append(effect_info)
                
                # Détecter les effets modifiés
                for effect_name in initial_effects_index:
                    if effect_name in final_effects_index:
                        initial_effect = initial_effects_index[effect_name]
                        final_effect = final_effects_index[effect_name]
                        
                        # Comparer les propriétés
                        initial_props = initial_effect.get("properties", [])
                        final_props = final_effect.get("properties", [])
                        
                        # Si les propriétés sont différentes
                        if initial_props != final_props:
                            # Trouver les propriétés qui ont changé
                            changed_props = []
                            initial_props_dict = {}
                            final_props_dict = {}
                            
                            for prop in initial_props:
                                if " : " in prop:
                                    key, value = prop.split(" : ", 1)
                                    initial_props_dict[key] = value
                            
                            for prop in final_props:
                                if " : " in prop:
                                    key, value = prop.split(" : ", 1)
                                    final_props_dict[key] = value
                            
                            # Détecter les propriétés modifiées
                            for key in initial_props_dict:
                                if key in final_props_dict and initial_props_dict[key] != final_props_dict[key]:
                                    changed_props.append({
                                        "property": key,
                                        "old_value": initial_props_dict[key],
                                        "new_value": final_props_dict[key]
                                    })
                            
                            # Détecter les nouvelles propriétés
                            for key in final_props_dict:
                                if key not in initial_props_dict:
                                    changed_props.append({
                                        "property": key,
                                        "old_value": None,
                                        "new_value": final_props_dict[key]
                                    })
                            
                            if changed_props:
                                effect_info = {
                                    "item_ID": item_id,
                                    "nodeId": final_item.get("nodeId"),
                                    "isText": final_item.get("isText", False),
                                    "effect_name": effect_name,
                                    "piste": final_piste,
                                    "start": final_start,
                                    "end": final_end,
                                    "changed_properties": changed_props
                                }
                                differences["effects_modified"].append(effect_info)
        
    except Exception as e:
        print(f"❌ Erreur lors de la comparaison de timeline: {str(e)}") 
    
    # Construire l'observation textuelle
    observation_parts = []

    if differences["items_added"]:
        observation_parts.append(f"Added {len(differences['items_added'])} clip(s)")
    if differences["items_removed"]:
        observation_parts.append(f"Deleted {len(differences['items_removed'])} clip(s)")
    if differences["items_moved_time"]:
        observation_parts.append(f"Moved {len(differences['items_moved_time'])} clip(s) in time")
    if differences["items_moved_track"]:
        observation_parts.append(f"Changed {len(differences['items_moved_track'])} clip(s) track")
    if differences["items_trimmed"]:
        observation_parts.append(f"Trimmed duration of {len(differences['items_trimmed'])} clip(s)")
    
    if differences["texts_added"]:
        observation_parts.append(f"Added {len(differences['texts_added'])} text(s)")
    if differences["texts_removed"]:
        observation_parts.append(f"Deleted {len(differences['texts_removed'])} text(s)")
    if differences["texts_moved_time"]:
        observation_parts.append(f"Moved {len(differences['texts_moved_time'])} text(s) in time")
    if differences["texts_moved_track"]:
        observation_parts.append(f"Changed {len(differences['texts_moved_track'])} text(s) track")
    if differences["texts_trimmed"]:
        observation_parts.append(f"Trimmed duration of {len(differences['texts_trimmed'])} text(s)")
    
    if differences["effects_added"]:
        observation_parts.append(f"Added {len(differences['effects_added'])} effect(s)")
    if differences["effects_modified"]:
        observation_parts.append(f"Modified {len(differences['effects_modified'])} effect(s)")
    
    observation = ", ".join(observation_parts) if observation_parts else "No changes detected"
    
    return differences, observation


class EditTimelineStructure(BaseModel):
    prompt: str = Field(description="""the creative task, the full edit to perform, the complete user intention. 
    """)
    include_effects :bool = Field(default=False, description="Use it only if you need to edit effects. Slower")
    include_metadata :bool = Field(default=False, description="Use it only if you need to know the description of the video and images to perform the task. Slower")

@tool("edit_timeline", args_schema=EditTimelineStructure)
async def edit_timeline(prompt: str, include_effects: bool = False, include_metadata: bool = False):
    """
    The Master Editing Agent for the active sequence.
    Capable of performing complex, multi-step editing tasks autonomously.
    
    CAPABILITIES:
    - Cook a complete edit from a single prompt
    - Modifiy an already existing edit. 
    - autonomous access to the timeline structure and project structure.

    USE THIS TOOL FOR:
    - "Make a vlog from these clips"
    - "Insert b-roll over the interview"
    - Any task requiring modification of the timeline content.
    """
    
    try:

        # Configuration de l'agent ReACT
        max_iterations = 20
        iteration = 0
        task_completed = False
        history = []
        
    
        # Do we need to include the project files in the context? 
        structured_output = {
            "type": "boolean",
            "description": "True if you need to include the project files in the context"
        }
        system_instruction = """
        You are a professional video editor. You receive a prompt from the user.
        Your task is to decide if you need to include the project files in the context to perform the task.
        True if you need access to insert items
        False if you only perform action directly on the timeline
        """.strip()

        include_project_files = gemini_call(f"Prompt: {prompt}", system_instruction, structured_output, None, 0.2, MODEL_REACT_PROJECT_STRUCTURE_TOOL)

        
        # Boucle ReACT
        print(f"\n{'='*80}") 
        while not task_completed and iteration < max_iterations:
            # Vérifier si l'utilisateur a demandé l'arrêt
            if config.STOP_REQUESTED:
                print("\n🛑 Arrêt demandé par l'utilisateur")
                config.API_STATUS = "End"
                return f"OBSERVATION: Task stopped by user at iteration {iteration}. Progress so far: {len(history)} action(s) completed."
            
            iteration += 1
               
            print(f"🔄 ITERATION {iteration}/{max_iterations}")
            
            # === PHASE 1: REASONING ======================================================
            config.API_STATUS = f"Getting timeline & project context... (iteration {iteration})"
            
            current_timeline, FPS, width, height = await get_timeline_structure.ainvoke({"include_metadata": include_metadata, "include_effects": False})
            current_project = await get_project_structure.ainvoke({"include_metadata": include_metadata, "include_audio": True  })
            
            config.API_STATUS = f"Thinking... (iteration {iteration})"
            
            # Construire le contexte avec l'historique
            history_text = ""
            if history:
                for i, h in enumerate(history, 1):
                    history_text += f"\n**Iteration {i}:**\n"
                    history_text += f"- Reasoning: {h['reasoning']}\n"
                    history_text += f"- Actions performed: {len(h['actions'])} action(s)\n"
                    history_text += f"- Observation: {h['observation']}\n"
            
            what_is_missing = ""
            if history:
                what_is_missing = f"- what is messing from Validation Agent : {reasoning_phase_4}"
            
            project_files_text = ""
            if include_project_files:
                project_files_text = f"- available files in project : {json.dumps(current_project, indent=2)}"
                
            # Structured output pour le reasoning
            structured_output = {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your analysis and reasoning about what to do next"
                    },
                    "actions": {
                        "type": "array",
                        "description": "List of actions to perform. Use the tools defined in the tool_list.",
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
                    },
                },
                "required": ["reasoning", "actions"]
            }
            
            reasoning_prompt = f"""
                You have performed the following actions before:
                ### AGENT HISTORY:
                {history_text}
            
                ### USER ORIGINAL REQUEST:
                {prompt}

                ### CURRENT STATUS:
                - actual timeline : {json.dumps(current_timeline, indent=2)}
                {what_is_missing}
                {project_files_text}

                ### YOUR TASK:
                Analyze the current timeline state and decide the NEXT BATCH of actions to perform

                As a Lead Senior Editor, here is a few rules you know :
                ### RULES for a GOOD VIDEO EDITING:
                - you can not choose a specific track to insert an item, Premiere Pro will handle it automatically
                - using all the available clips is not mandatory
                - don't leave space without video between two video clips
                - if it's music based edit, you must follow the music tempo (downbeats). Downbeats will be given in the timeline structure. So you must insert the music first to get the downbeats
                - Use inPoint_source or outPoint_source only for a music or a speech to select the right moment to insert. In any other case, do not use it.
                """.strip()
            
            system_instruction = f"""
                You are the Lead Editor working on the active timeline.
                Your goal is to complete the USER REQUEST by planning and executing actions step by step.
                
                # ACTIONS AVAILABLE:
                {EDIT_TIMELINE_STRUCTURE_TOOL_LIST}

                After each batch of actions, you will receive observations about what is missing to complete the USER REQUEST.
                
                You must follow this rules in your ReAct framework. 
                
                ### RULES for REASONING:
                1. Break down complex tasks into multiple iterations
                2. Consider dependencies between actions
                """.strip()

            # Appel au LLM pour le reasoning

            # sauvegarder dans un .txt le reasoning et le system instruction
            # with open("iteration_{}.txt".format(iteration), "w") as f:
            #     f.write(reasoning_prompt + "\n\n" + system_instruction)



            decision = gemini_call(reasoning_prompt, system_instruction, structured_output, None, 0.3, config.MODEL_AGENT_NAME)
            # decision = gptoss_call(reasoning_prompt, system_instruction, structured_output, None, 0.3, "openai/gpt-oss-120b")


            liste_actions = decision.get('actions', [])
            reasoning = decision.get('reasoning', 'Task completed')
            
            if config.REASONING_QUEUE:
                print(reasoning)
                await config.REASONING_QUEUE.put(reasoning)

            # print(f"📋 Planned actions: {liste_actions}")
            

            
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
                You are a professional video editing assistant using the ReAct framework.
                Your task is to format the actions send by the Reasoning phase to the format expected by the JavaScript function.
                """.strip()
                
            liste_actions = gemini_call(action_prompt, system_instruction, None, EDIT_TIMELINE_STRUCTURE_TOOL_LIST, 0.3, MODEL_REACT_PROJECT_STRUCTURE_TOOL)
            
            config.API_STATUS = f"Executing actions... (iteration {iteration})"
            

            
            # réordonner les actions: add texte et modifier en dernier 
            liste_actions = sorted(liste_actions, key=lambda x: x["name"] in ["edit_effect", "add_text", "modify_text"])
            actions_js_batch = []
            result_apply_effect = ""
            for action in liste_actions:


                # Vérifier si l'utilisateur a demandé l'arrêt avant d'exécuter les actions
                if config.STOP_REQUESTED:
                    print("\n🛑 Arrêt demandé par l'utilisateur pendant l'exécution des actions")
                    config.API_STATUS = "End"
                    return f"OBSERVATION: Task stopped by user during action execution at iteration {iteration}. Progress so far: {len(history)} action(s) completed."

                
                action_name = action.get("name")
                
                # Si c'est un cas spécial qui nécessite un traitement Python
                if action_name in ["edit_effect", "add_text", "modify_text"]:
                    print(action)
                    # D'abord, exécuter le batch JS accumulé
                    if actions_js_batch:
                        call_id = str(uuid.uuid4())
                        script_arg = json.dumps(actions_js_batch)
                        script = f"$._MYFUNCTIONS.executeTimelineActionList({script_arg});"
                        
                        PENDING_JS_CALLS[call_id] = {
                            "args": {"script": script},
                            "result": None,
                            "status": "pending"
                        }
                        
                        timeout = 180
                        start_time = time.time()
                        while time.time() - start_time < timeout:
                            if call_id in PENDING_JS_CALLS and PENDING_JS_CALLS[call_id]["status"] == "completed":
                                del PENDING_JS_CALLS[call_id]
                                break
                            await asyncio.sleep(0.5)
                        
                        if call_id in PENDING_JS_CALLS:
                            del PENDING_JS_CALLS[call_id]
                        
                        actions_js_batch = []
                    

                    # Ensuite, traiter le cas spécial
                    if action_name == "edit_effect":
                        for effect in action["args"]["effects"]:
                            ID = effect["ID"]
                            # Retrouver le no de track associé à l'ID dans current_timeline
                            for item in current_timeline:
                                if item["ID"] == ID:
                                    piste = item["piste"]
                                    nodeId = item['nodeId']
                                    break

                            project_item = find_item_by_node_id(json.loads(current_project).get("children", []), nodeId)

                            effect_prompt = effect["prompt"]

                            result_apply_effect = await apply_effect(ID, effect_prompt, piste, width, height, project_item['width'], project_item['height'])
                    
                    elif action_name == "add_text":
                        await add_text(action["args"]['texts'])
                    
                    elif action_name == "modify_text":
                        ID = action["args"]["ID"]
                        for item in current_timeline:
                            if item["ID"] == ID:
                                piste = item["piste"]
                                break
                        
                        await modify_text(ID, piste, action["args"]["prompt"])
                
                else:
                    # Pour les autres actions, ajouter les infos manquantes et accumuler pour JS
                    if action_name == "delete_item":
                        ID = action["args"]["ID"]
                        for item in current_timeline:
                            if item["ID"] == ID:
                                piste_type = item["piste"].split(" ")[0]
                                piste_number = int(item["piste"].split(" ")[1])
                                break
                        action["args"]["piste_type"] = piste_type
                        action["args"]["piste_number"] = piste_number
                        
                        print(action)
                    
                    elif action_name == "move_item":
                        liste_TC = [x/FPS for x in range(int(FPS))]
                        ID = action["args"]["ID"]
                        for item in current_timeline:
                            if item["ID"] == ID:
                                nodeId = item["nodeId"]
                                piste_type = item["piste"].split(" ")[0]
                                piste_number = int(item["piste"].split(" ")[1])
                                break
                        action["args"]["nodeId"] = nodeId
                        action["args"]["piste_type"] = piste_type
                        action["args"]["piste_number"] = piste_number
                        
                        start_decimal = action["args"]["start"] % 1
                        start_decimal_inf = max(x for x in liste_TC if x <= start_decimal)
                        action["args"]["start"] = action["args"]["start"] - start_decimal + start_decimal_inf
                        
                        if action["args"].get("end"):
                            end_decimal = action["args"]["end"] % 1
                            end_decimal_inf = max(x for x in liste_TC if x <= end_decimal)
                            action["args"]["end"] = action["args"]["end"] - end_decimal + end_decimal_inf
                        
                        print(action)
                        
                    elif action_name == "insert_item":
                        # accorder le start et end en fonction des FPS
                        # liste_TC = config.FPS_MAPPING[FPS]
                        liste_TC = [x/FPS for x in range(int(FPS))]

                        
                        # récupérer le décimal de start et end 
                        start_decimal = action["args"]["start"] % 1
                        start_decimal_inf = max(x for x in liste_TC if x <= start_decimal)
                        action["args"]["start"] = action["args"]["start"] - start_decimal + start_decimal_inf

                        if action["args"].get("end"):
                            end_decimal = action["args"]["end"] % 1
                            end_decimal_inf = max(x for x in liste_TC if x <= end_decimal)
                            action["args"]["end"] = action["args"]["end"] - end_decimal + end_decimal_inf
                        
                        
                        print(action)
                        
                    elif action_name == "insert_item_batch":
                        print(action)
                        liste_TC = [x/FPS for x in range(int(FPS))]


                        liste_nodeId = actions["args"]["list_nodeId"]
                        liste_start = actions["args"]["list_start"]
                        end = actions["args"]["last_end"]

                        for i in range(min(len(liste_nodeId), len(liste_start))):
                            action = {
                                "action_name": "insert_item",
                                "args": {
                                    "nodeId": liste_nodeId[i],
                                    "start": liste_start[i],
                                }
                            }
                            if i == min(len(liste_nodeId), len(liste_start)) - 1:
                                action["args"]["end"] = end
                        
                            actions_js_batch.append(action)
                    
                    actions_js_batch.append(action)
            
            # Exécuter le dernier batch JS s'il reste des actions
            if actions_js_batch:
                call_id = str(uuid.uuid4())
                script_arg = json.dumps(actions_js_batch)
                script = f"$._MYFUNCTIONS.executeTimelineActionList({script_arg});"
                
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
            
            # === PHASE 3: OBSERVATION ====================================================
            config.API_STATUS = f"Observing changes... (iteration {iteration})"
            
            # Récupérer la timeline après les actions
            post_action_timeline, FPS, width, height = await get_timeline_structure.ainvoke({"include_metadata": include_metadata, "include_effects": False})
            differences, observation = compare_timeline_structures(current_timeline, post_action_timeline)
            
            print(f"📊 Observation: {observation}")
            
            # Ajouter à l'historique
            history.append({
                "reasoning": reasoning,
                "actions": liste_actions,
                "observation": observation,
                "differences": differences
            })
            
            
            # === PHASE 4: EST CE QUE LE USER PROMPT EST FINIT ?  ====================================================
            
            structured_output = {
                "type": "object",
                "properties": {
                    "task_completed": {
                        "type": "boolean",
                        "description": "True if the user's request is fully completed and no more actions are needed"
                    }, 
                    "reasoning": {
                        "type": "string",
                        "description": "Explaination of the decision. If False, explain what is missing or incorrect to the Acting Agent so it can correct it."
                    }
                },
                "required": ["task_completed", "reasoning"]
            } 


            system_instruction = """
            You are a professional video editor on Premiere Pro in a ReACT Agent framework. You are after the Reasoning and Acting phases.
            Your task is to decide if the USER PROMPT is fully completed. If True, the edit will stop here. If false, the agent will continue to perform the next batch of actions planned.
            You receive the USER PROMPT, the timeline before and after the actions, the list of actions performed and the observation of the actions.
            
            There is some exceptions you must follows:
            ### EXCEPTIONS:
            1 Due to framerate, the start and end time of the imported items are not always precise. Don't try to correct this.
            2 You can't have access to transition used or speed details. Don't try to correct this
            3 You cant'have access to applied effects (resizing, moving, scaling, colorimetry, opacity, etc...). You can ignore this and consider this as completed. Just check that the effects are in the LIST OF PERFORMED ACTIONS.
            
            """.strip()
            
            final_prompt = f"""
            USER PROMPT: {prompt}
            TIMELINE BEFORE: {current_timeline}
            TIMELINE AFTER: {post_action_timeline}
            LIST OF PERFORMED ACTIONS: {liste_actions}
            OBSERVATION (excluding effects, transitions, speed): {observation}
            LAST REASONING FROM AGENT : {reasoning}
            """.strip()
            
            result_phase_4 = gemini_call(final_prompt, system_instruction, structured_output, None, 0.2, MODEL_REACT_PROJECT_STRUCTURE_TOOL)
            # result_phase_4 = gptoss_call(final_prompt, system_instruction, structured_output, None, 0.2, "openai/gpt-oss-120b")
            task_completed = result_phase_4.get("task_completed", False)
            reasoning_phase_4 = result_phase_4.get("reasoning", "")


            print(f"Task completed ? : {reasoning_phase_4}")
            if task_completed:
                if iteration == 1 and not liste_actions:
                    return f"OBSERVATION: Task already completed. {reasoning_phase_4}"
                break
            
            
            
            
        
        # Fin de la boucle ReACT


        print(f"Task completed: {task_completed}")
        print(f"Reasoning: {reasoning_phase_4}")
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
        
        # final_result = "OBSERVATION: Timeline edited successfully. " + " | ".join(final_summary_parts)
        final_result = "OBSERVATION: Timeline edited successfully. " + json.dumps(post_action_timeline, indent=2)
        config.API_STATUS = "End"
        return final_result
        
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









