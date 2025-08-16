# import json
# from typing import TypedDict, Annotated, List
# import operator
# import time
 
# from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage, HumanMessage
# from langchain.tools import tool
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode

# # --- Ajouts pour la communication JS
# import uuid
# import asyncio
# from LIB.shared_state import JS_TOOL_CALLS
# # --- Fin ajouts


# from pydantic import BaseModel, Field
# from typing import List


# class VideoPlan(BaseModel):
#     """Définit la structure d'un plan de production vidéo."""
#     effect_list: List[str] = Field(description="effect list to be used")


# def initialize_tools(google_api_key: str):
#     """Initialise tous les outils avec la clé API."""


#     #1. Obtenir l'architecture du projet
#     @tool
#     def get_project_architecture(video_topic: str) -> str:
#         """
#         Get the architecture of the video project. All the bins, the stems, the video clips, audio clips, mogrt, available in the project. 
#         """
#         # Cette fonction est maintenant juste une "déclaration" pour le LLM.
#         # La logique d'appel JS est gérée dans run_agent_streaming.
#         return "Project architecture will be retrieved from Premiere Pro."

#     @tool
#     def get_creative_intention(video_topic: str) -> str:
#         """
#         Call this at the begining of the video editing process.
#         Emphasize the creative intention of the video project
#         """
        
#         # 1. Initialiser le LLM
#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7, google_api_key=google_api_key)
        
        
#         # 2. Mettre à jour le prompt pour qu'il corresponde à la structure demandée
#         prompt = f"""
#         Your are a creative video editor who perfectly masters the codes of video editing, 
#         and the creative trends of the moment. 
#         You receive a video topic: '{video_topic}'.
#         Your goal is to emphasize the creative intention of the video project, 
#         and act as a Artistic Director to deliver a creative vision for the video.
#         """
        
#         # 3. Appeler le modèle structuré pour obtenir un objet Python
#         creative_intent = llm.invoke(prompt)

#         # 4. Formater l'objet en une chaîne de caractères lisible pour l'agent
#         return creative_intent

#     @tool
#     def get_effect_list(video_topic: str) -> str:
#         """
#         Get the effects to use in this project matching the right creative vision.
#         """
        
#         # 1. Initialiser le LLM
#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7, google_api_key=google_api_key)
        
#         # CHANGEMENT : On lie le LLM à notre structure Pydantic
#         structured_llm = llm.with_structured_output(VideoPlan)
        
#         # 2. Mettre à jour le prompt pour qu'il corresponde à la structure demandée
#         prompt = f"""
#         Your are a creative video editor who perfectly masters the rules and codes of video editing.
#         Your mission is to choose between a list of effect the ones that you could fit the user intention: '{video_topic}'.
#         The list of effects is the following:
#         - title
#         - transitions
#         - dynamic zoom
#         - black & white sad effect
#         - chapter titles
#         - dynamic text
#         """
        
#         # 3. Appeler le modèle structuré pour obtenir un objet Python
#         effect_list = structured_llm.invoke(prompt)

#         # 4. Formater l'objet en une chaîne de caractères lisible pour l'agent
#         return effect_list

#     @tool
#     def get_media_project(scene_description: str) -> List[str]:
#         """
#         Give the list of media files available for this project, matching the user request. 
        
#         """
#         print(f"... Outil 'Recherche Média' en cours pour '{scene_description}' ...")
#         time.sleep(1)

#         return ["rush_volcan_01.mp4", "drone_glacier_22.mov", "aurore_boreale_05.mxf"]

#     @tool
#     def add_rush(clip_name: str, time_IN: float) -> str:
#         """Add a clip by his clip_name at the right time time_IN to the current sequence."""
#         print(f"... Outil 'add_rush' en cours pour la séquence '{clip_name}', {time_IN} ...")
#         time.sleep(1)
#         return f"Succès."
    
#     @tool
#     def add_sound(sound_name: str, time_IN: float) -> str:
#         """Add a sound or music by his sound_name at the right time time_IN to the current sequence."""
#         print(f"... Outil 'add_sound' en cours pour la séquence '{sound_name}', {time_IN} ...")
#         time.sleep(1)
#         return f"Succès."
    
#     # @tool
#     # def get_sequence_state() -> str:
#     #     """Récupère les informations de la séquence pour la séquence de montage en cours, dont le nom de la séquence active."""
#     #     print(f"... Outil 'Récupération de la Timeline' en cours ...")
#     #     time.sleep(1)
#     #     return "nom de la timeline : TESTA"
    
#     @tool
#     def add_effect(effect_name: str, time_IN: float) -> str:
#         """Add an effect by his effect_name at the right time time_IN to the current sequence."""
#         print(f"... Outil 'Ajout d'Effet' en cours pour l'effet '{effect_name} au temps {time_IN}' ...")
#         time.sleep(1)
#         return f"Effet '{effect_name}' ajouté avec succès."

#     # return [get_effect_list, get_media_project, get_creative_intention, add_rush, add_sound, add_effect]
#     return [get_project_architecture]

# TOOL_DISPLAY_MAPPING = {
#     "get_creative_intention": {
#         "title": "Getting the creative intention",
#         "category": "creative"
#     },
#     "get_project_architecture": {
#         "title": "Analyzing Project Structure",
#         "category": "project"
#     },
#     "get_media_project": {
#         "title": "Searching for media",
#         "category": "video"
#     },
#     "add_rush": {
#         "title": "Adding footage",
#         "category": "video"
#     },
#     "add_sound": {
#         "title": "Adding sound design",
#         "category": "sound"
#     },
#     "add_effect": {
#         "title": "Adding effects",
#         "category": "effect"
#     }
# }
 
# # --- 2. Définition du Graphe (inchangé) ---
# class AgentState(TypedDict):
#     messages: Annotated[list, operator.add]
 
# def create_agent_graph(google_api_key: str):

#     tools = initialize_tools(google_api_key)
#     tool_node = ToolNode(tools)
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5, google_api_key=google_api_key)
#     model = model.bind_tools(tools)
    
#     def agent(state: AgentState):
#         response = model.invoke(state["messages"])
#         return {"messages": [response]}
    
#     def should_continue(state: AgentState) -> str:
#         if not state["messages"][-1].tool_calls:
#             return "end"
#         return "continue"
    
#     workflow = StateGraph(AgentState)
#     workflow.add_node("agent", agent)
#     workflow.add_node("tools", tool_node)
#     workflow.set_entry_point("agent")
#     workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
#     workflow.add_edge("tools", "agent")
#     return workflow.compile()

# async def run_agent_streaming(user_input: str, google_api_key: str):
#     """Exécute l'agent et streame les résultats sous forme de JSON."""
#     app = create_agent_graph(google_api_key)

#     conversation_history = [
#         SystemMessage(content="""You are a helpful video editing assistant. 
#                       Your task is to call the tool `get_project_architecture` and answer to the user with that"""),
#         HumanMessage(content=user_input)
#     ]

#     # Boucle pour gérer les invocations successives de l'agent
#     while True:
#         inputs = {"messages": conversation_history}
        
#         # On stocke le dernier message pour savoir si on doit continuer
#         last_message = None

#         async for chunk in app.astream(inputs, stream_mode="values"):
#             last_message = chunk["messages"][-1]
            
#             if isinstance(last_message, AIMessage):
#                 if last_message.tool_calls:
#                     # L'agent a décidé d'appeler un outil
#                     if last_message.content:
#                         yield {"type": "thought", "content": f"🤖 {last_message.content}"}
                    
#                     tc = last_message.tool_calls[0]
#                     tool_info = TOOL_DISPLAY_MAPPING.get(tc['name'], {"title": tc['name'], "category": "default"})
                    
#                     if tc['name'] == 'get_project_architecture':
#                         # On intercepte pour appeler notre outil JS
#                         conversation_history.append(last_message) # On sauvegarde la décision de l'agent

#                         call_id = str(uuid.uuid4())
#                         event = asyncio.Event()
#                         JS_TOOL_CALLS[call_id] = {"event": event}

#                         # On notifie le client pour qu'il exécute la fonction JS
#                         yield {
#                             "type": "js_tool_call",
#                             "tool_name": "getProjectArchitecture",
#                             "call_id": call_id,
#                             "args": {},
#                             "title": tool_info["title"],
#                             "category": tool_info["category"]
#                         }
                        
#                         await event.wait()
#                         result = JS_TOOL_CALLS[call_id].get('result')
#                         del JS_TOOL_CALLS[call_id]
                        
#                         # On ajoute le résultat à l'historique
#                         tool_message = ToolMessage(content=str(result), tool_call_id=tc['id'])
#                         conversation_history.append(tool_message)
                        
#                         # On sort de la boucle de streaming pour relancer l'agent
#                         break 
#                     else:
#                         # Gérer les outils Python normaux ici si nécessaire
#                         yield { "type": "tool_start", "title": tool_info["title"], "category": tool_info["category"], "args": tc['args'] }
#                 else:
#                     # C'est la réponse finale de l'agent
#                     yield {"type": "answer", "content": last_message.content}
            
#             elif isinstance(last_message, ToolMessage):
#                 pass
        
#         # Condition de sortie de la boucle `while`
#         if last_message and isinstance(last_message, AIMessage) and not last_message.tool_calls:
#             # L'agent a donné une réponse finale, on peut arrêter.
#             break
        
#         # Sécurité pour éviter une boucle infinie
#         if not last_message:
#             break