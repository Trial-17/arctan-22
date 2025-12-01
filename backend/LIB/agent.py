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
    "edit_timeline": {
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
    "get_to_do_list_for_timeline_structure": {
        "title": "Crafting todo",
        "category": "extendscript"
    },
    "get_creative_todo": {
        "title": "Crafting todo",
        "category": "extendscript"
    }, 
    "export_sequence": {
        "title": "Exporting sequence",
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
        
        log_gemini_call("agent.py::gemini_call", payload, result)  # LOG
        
        # Retourner le résultat dans le même format que l'ancienne fonction
        return result
        
    except requests.exceptions.RequestException as e:
        log_gemini_call("agent.py::gemini_call", payload if 'payload' in locals() else {}, None, str(e))  # LOG
        print(f"❌ Erreur lors de l'appel à l'API relai: {str(e)}")
        raise Exception(f"Erreur lors de l'appel à l'API relai: {str(e)}")





# ------------- AGENT -------------


class GetToDoListForProjectStructure(BaseModel):
    prompt: str = Field(description="the prompt to get the to do list for the project structure")

@tool("get_to_do_list_for_project_structure", args_schema=GetToDoListForProjectStructure)
async def get_to_do_list_for_project_structure(prompt: str):
    """
    Ask the to do list to handle the user prompt if it's focused on the project structure
    """
    
    
    system_instruction = f"""
    You are a professional video editor.
    You receive a user prompt.
    Your task is to think how to handle the user prompt, and return a to do list to describe to an AI agent how to handle the user prompt.
    
    
    ### TOOLS access:
    - get_project_structure : give informations about available media to use
    - edit_project_structure : use to create sequences, bins, organize the project structure. Can handle multiple actions in one call.
    - labelize_audio : use to get music tempo, transcription of audio before using it 
    
    
    you can use the followings common rules about video editing : 
    
    ### RULES:
    - if needed specifiy the tool to use in your todo point
    - for COMPLEX tasks with many operations, BREAK THEM DOWN into multiple tool calls instead of one big call.

    
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



class GetCreativeTodo(BaseModel):
    prompt: str = Field(description="the prompt to get the creative todo")

@tool("get_creative_todo", args_schema=GetCreativeTodo)
async def get_creative_todo(prompt: str):
    """
    Get the creative todo to handle the user prompt if it's focus on creative tasks such a new edit
    """
    
    
    system_instruction = f"""
    You are a professional video editor.
    You receive a user prompt.

    ### TASK 
    1. Analyze the user prompt and understand the user creative intention
    2. Cook a creative intention to perform the user prompt
    3. return a to do list to describe to an AI agent how to handle the user prompt, use following tools
    
    
    
    ### TOOLS access:
    - get_project_structure : give informations about available media to use
    - get_timeline_structure : use to get the structure of a timeline
    - edit_timeline_structure : use to add clips, audio, effects, edit a timeline etc...
    - open_timeline : use to open a timeline before editing it or grepping it
    - labelize_audio : use to get music tempo, transcription of audio before using it 
    
    
    you can use the followings common rules about video editing : 
    
    ### EXEMPLES: 
    - SOCIAL MEDIA EDIT: define audience, duration, style, music

    
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
             get_timeline_structure, edit_timeline, 
             open_timeline,
             labelize_audio,
             get_creative_todo, 
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
    app = create_agent_graph(model)

    # --- Gestion de l'historique ---
    conversation_history = [
        SystemMessage(content="""
        You are an expert Video Editing Orchestrator for Premiere Pro.
        Your role is to understand the user's request and delegate the work to the appropriate specialized agent.

        ### YOUR TEAM OF EXPERTS:
        1.  **edit_project_structure**: Expert in organizing the project bin. Use it to create sequences, bins, or organize items.
        2.  **edit_timeline**: The MASTER EDITOR. Capable of handling FULL editing tasks. It can add clips, music, text, effects, and adjust timing ALL IN ONE SESSION. He has his own access to get project structure and timeline structure. So no Need to send it to the agent.
        3.  **get_project_structure** / **get_timeline_structure**: Analysts to retrieve information when you are unsure.
        4.  **export_sequence**: Finalizer to export the result. 

        ### KEY RULES FOR DELEGATION:
        - **DO NOT MICROMANAGE:** If the user wants a video edit, delegate the ENTIRE task to `edit_timeline` in a single call. Do not split it into "add video" then "add audio". The `edit_timeline` agent is smart enough to handle the whole flow.
        - **PASS CONTEXT:** When calling a tool, ensure the prompt contains all necessary creative details (mood, style, pacing).
        - **VERIFY:** After a tool finishes, check the output to ensure the user's request is met.

        ### EXAMPLES:

        #### Example 1: Creating a Reel 
        **User:** "Create a dynamic reel from the 'Travel' bin with upbeat music and subtitles."
        **You:**
        1.  Call `get_project_structure` to see available media. And labelize required audio if needed
        2.  Call `edit_project_structure` to create a sequence "Travel_Reel" (1080x1920).
        3.  Call `edit_timeline` with prompt: "Create a dynamic reel using clips from 'Travel' bin. Sync to upbeat music and add subtitles." (DELEGATE EVERYTHING)


        #### Example 2: Create a Movie
        **User:** "create a cinmeatic film with my rush"
        **You:**
        1.  Call `get_project_structure` to see available media and audio to use
        2.  Call `edit_project_structure` to create a sequence cinematic.
        3.  Call `edit_timeline`
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

    async for chunk in app.astream_events(inputs, version="v1"):
        if config.STOP_REQUESTED:
            print("🛑 Agent stopped by user request.")
            yield {"type": "thought", "content": "\n[Agent stopped by user]"}
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
            
            if tool_name not in ["get_project_structure", "get_timeline_structure"]:
                print(f"\n--- [TOOL RESULT] --- - From: `{tool_name}` - Output: {output_preview}")

            # Diminuer la profondeur et n'émettre l'événement de fin que pour le top-level
            tool_call_depth = max(0, tool_call_depth - 1)
            if tool_call_depth == 0:
                tool_info = TOOL_DISPLAY_MAPPING.get(tool_name, {"title": tool_name, "category": "default"})
                yield {
                    "type": "tool_end",
                    "title": tool_info["title"]
                }
