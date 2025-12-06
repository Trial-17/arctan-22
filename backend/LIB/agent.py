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
from LIB.AGENT_project import get_project_structure, edit_project_structure, labelize_audio
from LIB.AGENT_timeline import get_timeline_structure, edit_timeline, open_timeline
from LIB.AGENT_export import export_sequence

MODEL_CONTEXT_1 = "gemini-2.5-flash-lite" # Get Project Context
MODEL_CONTEXT_2 = "gemini-2.5-flash-lite" # Get Timeline Context
MODEL_TOOL_1 = "gemini-2.5-flash-lite" # Edit project structure
MODEL_TOOL_2 = "gemini-2.5-flash-lite" # Choose Effect properties



TOOL_DISPLAY_MAPPING = {
    "get_project_structure": {
        "titles": ["Grepping project", "Reading project", "Analyzing project", "Scanning project"],
        "category": "extendscript",
        "icon": "agent_project_read.png"
    },
    "edit_project_structure": {
        "titles": ["Editing project", "Modifying project", "Updating project", "Restructuring project"],
        "category": "extendscript",
        "icon": "agent_project_edit.png"
    },
    "get_timeline_structure": {
        "titles": ["Grepping timeline", "Reading timeline", "Analyzing timeline", "Scanning timeline"],
        "category": "extendscript",
        "icon": "agent_timeline_read.png"
    },
    "edit_timeline": {
        "titles": ["Editing timeline", "Modifying timeline", "Updating timeline", "Crafting timeline"],
        "category": "extendscript",
        "icon": "agent_timeline_edit.png"
    }, 
    "labelize_audio": {
        "titles": ["Grepping audio", "Analyzing audio", "Processing audio", "Scanning audio"],
        "category": "extendscript",
        "icon": "agent_audio.png"
    },
    "open_timeline": {
        "titles": ["Opening timeline", "Loading timeline", "Accessing timeline", "Switching timeline"],
        "category": "extendscript",
        "icon": "agent_open.png"
    }, 
    "get_creative_todo": {
        "titles": ["Crafting todo", "Planning workflow", "Designing approach", "Organizing steps"],
        "category": "extendscript",
        "icon": "agent_project_read.png"
    }, 
    "export_sequence": {
        "titles": ["Exporting sequence", "Rendering sequence", "Outputting sequence", "Generating export"],
        "category": "extendscript",
        "icon": "agent_render.png"
    }
}

def get_random_tool_title(tool_name):
    """Sélectionne aléatoirement un titre parmi les variantes disponibles pour un outil."""
    import random
    tool_info = TOOL_DISPLAY_MAPPING.get(tool_name, {"titles": [tool_name], "category": "default", "icon": "check.png"})
    return random.choice(tool_info["titles"])


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





# ------------- AGENT -------------


class GetCreativeTodo(BaseModel):
    prompt: str = Field(description="the user prompt")

@tool("get_creative_todo", args_schema=GetCreativeTodo)
async def get_creative_todo(prompt: str):
    """
    Call this tool before performing the user prompt.
    It will give you a todo to follow to handle the user prompt.
    Don't call it if it's simply answering a question.

    """
    
    try: 
        system_instruction = f"""
        You are a professional Video Editor Planificator.
        You have diffrent tools at your disposal to handle the user prompt.


        Your task :
        ### TASK 
        1. Analyze the user prompt and understand the user creative intention
        2. return a to do list to describe to an AI agent how to handle the user prompt, using following tools and constraints
        
        ### TOOLS ARCHITECTURE:
        - get_project_structure : return the structure of the Premiere Pro project. Usefull to get the availables media and project architecture
        - labelize_audio : return the downbeats or transcription of a media. Provide de media name and audio type (music or speech). 1 call per media
        - edit_project_structure : this is a ReACT agent, use it to create sequences (also named timelines), bins, rename, move items, organize the project structure. Can handle multiple actions in one call. This tool has his own access to get_project_structure.
            - edit_project_structure TOOLS (using nodeId to identify the item to modify or parent_nodeId to identify the path) : 
                    - create_bin_tool
                    - delete_bin_tool (only works with empty bins)
                    - create_sequence_tool (width, height, framerate)
                    - update_sequence_tool (width, height, framerate)
                    - clone_sequence_tool (usefull to creat backups, new versions, alternative ideas)
                    - modify_item_tool (rename, move anything)
                    - move_batch_tool (move multiple items at once)
                END of edit_project_structure TOOLS

        - open_timeline : open a timeline (also named a sequence) before editing it or getting its structure. The others tools only have access to the opened timeline (also named a sequence).
        - get_timeline_structure : return the structure of the opened timeline (also named a sequence). Usefull to get the edit architecture.
        - edit_timeline : this is a ReACT agent, use it to edit the opened timeline (also named a sequence). Can handle multiple actions in one call. This tool has his own access to get_timeline_structure and get_project_structure.
                - edit_timeline TOOLS (using ID which is the ID of the itme on the timeline)
                    - insert_item_tool
                    - insert_item_batch_tool (usefull for adding multiple items on music beats)
                    - move_item_tool
                    - delete_item_tool
                    - add_marker_tool
                    - edit_effect_tool (only lumetri, opacity, position, scale, crop)
                    - add_text_tool
                    - modify_text_tool
                END of edit_timeline TOOLS

        - export_sequence : export the opened timeline (also named a sequence) to a file. It will determinate by itself the export format.


        ### OUTPUT
        - describe the step and intention. The tool will handle the parametrization.
        - Export only if asked by the user.

        """
        
        structured_output = {
            "type": "array",
            "items": {
                "type": "string",
                "description": "The precise description of the step to perform"
            }
        }



        previsous_todo = {}



        final_prompt = f"""



        ### NEW USER PROMPT
        {prompt}
        """
        
        todo_list = gemini_call(final_prompt, system_instruction, structured_output, None, 0.2, config.MODEL_AGENT_NAME)
        
        print(f"Todo list")
        for todo in todo_list:
            print(f"- {todo}")
        
        return todo_list
    except Exception as e:
        print(f"❌ Error in get_creative_todo: {str(e)}")
        return None





class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
 
def create_agent_graph(model_name: str = "fast"):

    tools = [get_project_structure, edit_project_structure, 
             get_timeline_structure, edit_timeline, 
             open_timeline,
             labelize_audio,
            #  get_creative_todo, 
             export_sequence,
             ]
    
    tool_node = ToolNode(tools)
    if model_name == "fast": config.MODEL_AGENT_NAME = "gemini-2.5-flash"
    else : config.MODEL_AGENT_NAME = "gemini-2.5-pro"

    
    model = PremiereGPT_LLM(model=model_name, temperature=0.5)
    model = model.bind_tools(tools)
    
    def agent(state: AgentState):
        if config.STOP_REQUESTED:
            return {"messages": [AIMessage(content="Agent stopped by user.")]}
        
        # Vérifier s'il y a des tool calls en attente dans la file d'attente
        if hasattr(config, 'PENDING_TOOL_CALLS_QUEUE') and config.PENDING_TOOL_CALLS_QUEUE:
            # Prendre le premier tool call de la file d'attente
            next_call = config.PENDING_TOOL_CALLS_QUEUE.pop(0)
            print(f"🔄 Exécution du tool en file d'attente: {next_call['name']} (restants: {len(config.PENDING_TOOL_CALLS_QUEUE)})")
            
            # Créer un AIMessage avec le tool call
            tool_call_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": next_call["name"],
                    "args": next_call["args"],
                    "id": f"call_{next_call['name']}_{hash(str(next_call['args']))}"
                }],
                additional_kwargs={},
                response_metadata={"model_name": model.model_name}
            )
            return {"messages": [tool_call_message]}
        
        # Sinon, appeler le LLM normalement
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> str:
        if config.STOP_REQUESTED:
            return "end"
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
    # Définir le token global pour cette session
    config.AGENT_TOKEN = token
    
    # Initialiser la file d'attente des tool calls pour cette nouvelle session
    config.PENDING_TOOL_CALLS_QUEUE = []

    # Reset history if new conversation
    if not existing_history:
        config.COPILOT_HISTORY.reset()

    config.COPILOT_HISTORY.add("message User", user_input)
    
    app = create_agent_graph(model)

    # --- Gestion de l'historique ---
    conversation_history = [
        SystemMessage(content="""
        You are an expert Video Editing Orchestrator for Premiere Pro.
        Your task is to understand the user's request and delegate the work to the appropriate specialized agent.

        ### TOOLS ARCHITECTURE:

        - get_project_structure : return the structure of the Premiere Pro project. Usefull to get the availables media and project architecture
        - labelize_audio : return the downbeats or transcription of a media. Provide de media name and audio type (music or speech). 1 call per media
        - edit_project_structure : this is a ReACT agent, use it to create sequences, bins, rename, move items, organize the project structure. Can handle multiple actions in one call. This tool has his own access to get_project_structure.
            - edit_project_structure TOOLS (using nodeId to identify the item to modify or parent_nodeId to identify the path) : 
                    - create_bin_tool
                    - delete_bin_tool (only works with empty bins)
                    - create_sequence_tool (width, height, framerate)
                    - update_sequence_tool (width, height, framerate)
                    - clone_sequence_tool (usefull to creat backups, new versions, alternative ideas)
                    - modify_item_tool (rename, move anything)
                    - move_batch_tool (move multiple items at once)
                END of edit_project_structure TOOLS

        - open_timeline : open a timeline (also named a sequence) before editing it or getting its structure. The others tools only have access to the opened timeline (also named a sequence).
        - get_timeline_structure : return the structure of the opened timeline (also named a sequence). Usefull to get the edit architecture.
        - edit_timeline : this is a ReACT agent, use it to edit the opened timeline (also named a sequence). Can handle multiple actions in one call. This tool has his own access to get_timeline_structure and get_project_structure.
                - edit_timeline TOOLS (using ID which is the ID of the itme on the timeline)
                    - insert_item_tool
                    - insert_item_batch_tool (usefull for adding multiple items on music beats)
                    - move_item_tool
                    - delete_item_tool
                    - add_marker_tool
                    - edit_effect_tool (only lumetri, opacity, position, scale, crop)
                    - add_text_tool
                    - modify_text_tool
                END of edit_timeline TOOLS

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
    

    config.AGENT_HISTORY = conversation_history
    
    # --------------------------------


    inputs = {"messages": conversation_history}
    
    thought_accumulator = ""
    tool_call_depth = 0  # profondeur d'appel des tools; 0 = appel top-level
    tool_titles = {}  # Stocke les titres choisis aléatoirement pour chaque outil

    async for chunk in app.astream_events(inputs, version="v1"):
        if config.STOP_REQUESTED:
            print("🛑 Agent stopped by user request.")
            yield {"type": "thought", "content": "\n[Agent stopped by user]"}
            if thought_accumulator.strip():
                config.COPILOT_HISTORY.add("message renvoyé", thought_accumulator.strip())
            break

        kind = chunk["event"]
        
        if kind == "on_chat_model_stream":
            content = chunk["data"]["chunk"].content
            if content:
                # Conserver exactement le contenu streamé (y compris \n et espaces d'indentation)
                thought_accumulator += content
                print(content)
                yield {"type": "thought", "content": content}
                await asyncio.sleep(0.01)

        elif kind == "on_tool_start":
            tool_name = chunk["name"]
            tool_args = chunk["data"].get("input")
            if tool_name not in ["get_project_structure", "get_timeline_structure"]:
                print(f"\n--- [TOOL CALL] ---" + f"  - Name: `{tool_name}`" + f"  - Arguments: {tool_args}")

            # Afficher uniquement pour les appels top-level (pas ceux déclenchés par un autre tool)
            if tool_call_depth == 0:
                # Log reflection before tool call
                if thought_accumulator.strip():
                    config.COPILOT_HISTORY.add("reflexion", thought_accumulator.strip())
                    thought_accumulator = ""

                config.COPILOT_HISTORY.add("tool appele", {"name": tool_name, "args": tool_args})

                tool_info = TOOL_DISPLAY_MAPPING.get(tool_name, {"titles": [tool_name], "category": "default", "icon": "check.png"})
                chosen_title = get_random_tool_title(tool_name)
                tool_titles[tool_name] = chosen_title  # Stocker le titre pour tool_end
                
                yield {
                    "type": "tool_start",
                    "title": chosen_title,
                    "category": tool_info["category"],
                    "icon": tool_info.get("icon", "check.png"),
                    "args": tool_args
                }

            tool_call_depth += 1

        elif kind == "on_tool_end":
            tool_name = chunk["name"]
            tool_output = chunk["data"].get("output")
            
            output_preview = (str(tool_output)[:300] + '...') if len(str(tool_output)) > 300 else str(tool_output)
            output_preview = output_preview.replace('\n', ' ').replace('\r', ' ').replace('    ', ' ').replace('  ', ' ')
            
            if tool_name not in ["get_project_structure", "get_timeline_structure"]:
                print(f"\n--- [TOOL RESULT] --- - From: `{tool_name}` - Output: {output_preview}")

            # Diminuer la profondeur et n'émettre l'événement de fin que pour le top-level
            tool_call_depth = max(0, tool_call_depth - 1)
            if tool_call_depth == 0:
                config.COPILOT_HISTORY.add("tool result", {"name": tool_name, "result": tool_output})

                tool_info = TOOL_DISPLAY_MAPPING.get(tool_name, {"titles": [tool_name], "category": "default"})
                # Récupérer le titre stocké lors du tool_start
                chosen_title = tool_titles.get(tool_name, get_random_tool_title(tool_name))
                yield {
                    "type": "tool_end",
                    "title": chosen_title
                }

    if thought_accumulator.strip():
        config.COPILOT_HISTORY.add("message renvoyé", thought_accumulator.strip())
