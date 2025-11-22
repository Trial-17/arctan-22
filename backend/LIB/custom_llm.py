from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

import json
import inspect
import json
from typing import List
from langchain_core.tools import StructuredTool
import re
import requests
import ast
from LIB.gemini_logger import log_gemini_call


def clean_prompt(text: str) -> str:
    # Supprimer les espaces en début et fin de ligne
    text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)
    # Remplacer plusieurs espaces consécutifs par un seul
    text = re.sub(r' {2,}', ' ', text)
    # Supprimer les lignes vides multiples
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()


def fix_gemini_response(response, tool_list=None):
    """
    Vérifie et corrige le format de la réponse Gemini.
    
    Détecte si la réponse est mal formatée (texte décrivant des actions au lieu d'une liste)
    et la convertit au bon format.
    
    Args:
        response: La réponse brute de Gemini (liste d'actions ou texte)
        tool_list: Liste des tools disponibles (pour validation)
    
    Returns:
        La réponse corrigée au bon format
    """
    
    try : 
        # Si c'est déjà une liste de dictionnaires avec name et args, c'est bon
        if isinstance(response, list) and all(isinstance(item, dict) and 'name' in item and 'args' in item for item in response):
            return response
        
        # Si c'est une chaîne qui ressemble à une description de tool calls
        if isinstance(response, str):
            # Pattern 1: "AI Tool calls: ['Tool call: action_name args: {...}']"
            pattern1 = r"Tool call:\s*(\w+)\s+args:\s*(\{[^}]+\})"
            matches1 = re.findall(pattern1, response)
            
            if matches1:
                tool_calls = []
                for tool_name, args_str in matches1:
                    try:
                        # Nettoyer les échappements de quotes
                        args_str = args_str.replace("\\'", "'").replace('\\"', '"')
                        # Parser les arguments (format Python ou JSON)
                        try:
                            # Essayer d'abord ast.literal_eval pour le format Python
                            args = ast.literal_eval(args_str)
                        except (ValueError, SyntaxError):
                            # Si ça échoue, essayer json.loads pour le format JSON
                            args = json.loads(args_str)
                        tool_calls.append({
                            "name": tool_name,
                            "args": args
                        })
                    except Exception as e:
                        print(f"⚠️ Erreur lors du parsing des args pour {tool_name}: {e}")
                        continue
                
                if tool_calls:
                    print(f"✅ Correction automatique: {len(tool_calls)} tool call(s) détecté(s) et corrigé(s)")
                    return tool_calls
            
            # Pattern 2: Recherche plus flexible pour "action_name(...)"
            pattern2 = r"(\w+)\s*\(([^)]+)\)"
            matches2 = re.findall(pattern2, response)
            
            if matches2 and tool_list:
                # Vérifier que les noms correspondent à des tools disponibles
                available_tools = [tool['name'] for tool in tool_list] if tool_list else []
                tool_calls = []
                
                for tool_name, args_str in matches2:
                    if tool_name in available_tools:
                        try:
                            # Essayer de parser comme JSON ou comme key=value
                            if '{' in args_str:
                                try:
                                    # Essayer d'abord ast.literal_eval pour le format Python
                                    args = ast.literal_eval(args_str)
                                except (ValueError, SyntaxError):
                                    # Si ça échoue, essayer json.loads pour le format JSON
                                    args = json.loads(args_str)
                            else:
                                # Parser format key=value
                                args = {}
                                for pair in args_str.split(','):
                                    if '=' in pair:
                                        key, value = pair.split('=', 1)
                                        key = key.strip().strip("'\"")
                                        value = value.strip().strip("'\"")
                                        args[key] = value
                            
                            tool_calls.append({
                                "name": tool_name,
                                "args": args
                            })
                        except Exception as e:
                            print(f"⚠️ Erreur lors du parsing des args pour {tool_name}: {e}")
                            continue
                
                if tool_calls:
                    print(f"✅ Correction automatique (pattern 2): {len(tool_calls)} tool call(s) détecté(s) et corrigé(s)")
                    return tool_calls
        
    except Exception as e:
        print(f"❌ Erreur lors de la correction du format de la réponse: {e}")
        return response
    return response


def gemini_call(messages, system_instruction, structured_output = None, tool_list = None, temperature= 0.1,  model = "fast"):
    """
    Appelle l'API Gemini via l'API relai.
    Conserve la même interface et le même comportement que l'ancienne fonction.
    """
    from LIB import config
    
    # Vérifier le token d'authentification
    if not config.AGENT_TOKEN:
        raise Exception("Aucun token d'authentification disponible pour l'appel API")
    
    try:

        
        # Convertir les messages LangChain en format dictionnaire
        messages_dict = []
        for message in messages:
            if isinstance(message, SystemMessage):
                messages_dict.append({
                    "type": "system",
                    "content": clean_prompt(message.content)
                })
            elif isinstance(message, HumanMessage):
                messages_dict.append({
                    "type": "human",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                msg_dict = {
                    "type": "ai",
                    "content": message.content
                }
                # Ajouter les tool_calls si présents
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    msg_dict["tool_calls"] = message.tool_calls
                messages_dict.append(msg_dict)
            elif isinstance(message, ToolMessage):
                messages_dict.append({
                    "type": "tool",
                    "content": message.content
                })
        
        # Préparer la requête pour l'API relai
        payload = {
            "messages": messages_dict,
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
            f"{config.API_URL}/agent",
            json=payload,
            headers={"Authorization": f"Bearer {config.AGENT_TOKEN}"}
        )
        
        # Vérifier le statut de la réponse
        if response.status_code != 200:
            raise Exception(f"Erreur API (status {response.status_code}): {response.text}")
        
        # Extraire le résultat
        result = response.json().get("result")
        
        log_gemini_call("custom_llm.py::gemini_call", payload, result)  # LOG
        
        # Retourner le résultat dans le même format que l'ancienne fonction
        return result
        
    except requests.exceptions.RequestException as e:
        log_gemini_call("custom_llm.py::gemini_call", payload if 'payload' in locals() else {}, None, str(e))  # LOG
        print(f"❌ Erreur lors de l'appel à l'API relai: {str(e)}")
        raise Exception(f"Erreur lors de l'appel à l'API relai: {str(e)}")




class PremiereGPT_LLM(BaseChatModel):
    """
    Custom LLM by PremiereGPT
    """

    model_name: str = Field(alias="model")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2
    tools: Optional[List[Dict]] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # # Replace this with actual logic to generate a response from a list

                
        # print("[CUSTOM LLM]--------------------------------")
        # print(f"Messages: {messages}")
        response = gemini_call(messages, None, None, self.tools, self.temperature, self.model_name)
        
        # Corriger le format de la réponse si nécessaire
        response = fix_gemini_response(response, self.tools)
        
        # print(f"Response: {response}")
        
        # Si c'est une liste de function calls, créer un AIMessage avec tool_calls
        if isinstance(response, list) and response:
            # Convertir les function calls au format LangChain
            tool_calls = []
            for call in response:
                tool_calls.append({
                    "name": call["name"],
                    "args": call["args"],
                    "id": f"call_{call['name']}_{hash(str(call['args']))}"  # ID unique
                })
            
            message = AIMessage(
                content="",  # Contenu vide pour les tool calls
                tool_calls=tool_calls,
                additional_kwargs={},
                response_metadata={
                    "model_name": self.model_name,
                },
            )
        else:
            # Réponse textuelle normale
            message = AIMessage(
                content=response,
                additional_kwargs={},
                response_metadata={
                    "model_name": self.model_name,
                },
            )
        ##

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """

        """
        
        # print("[CUSTOM LLM]--------------------------------")
        # print(f"Prompt: {prompt}")
        # Appel classique de la génération (sans streaming réel)
        full_output = self._generate(
            prompt,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )
        
        # print("[CUSTOM LLM]--------------------------------")
        # print(f"Full Output: {full_output}")
        
        # Récupérer le message de la réponse
        message = full_output.generations[0].message
        
        # Créer un chunk avec le message complet (texte ou tool_calls)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Pour les tool calls, créer un chunk avec tool_calls
            message_chunk = AIMessageChunk(
                content=message.content,
                tool_calls=message.tool_calls
            )
        else:
            # Pour le texte normal
            message_chunk = AIMessageChunk(content=message.content)
        
        chunk = ChatGenerationChunk(message=message_chunk)

        # Envoyer le token final au gestionnaire de callback s'il est présent
        if run_manager:
            run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)

        # Yielder le chunk unique
        yield chunk

    def bind_tools(self, tools):


        def flatten_schema(schema: dict) -> dict:
            """Remplace les $ref et $defs par la définition inline."""
            if "$defs" in schema:
                defs = schema.pop("$defs")
            else:
                defs = {}

            def replace_refs(obj):
                if isinstance(obj, dict):
                    if "$ref" in obj:
                        ref = obj.pop("$ref")
                        key = ref.split("/")[-1]
                        return replace_refs(defs[key])
                    else:
                        return {k: replace_refs(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_refs(v) for v in obj]
                else:
                    return obj

            return replace_refs(schema)


        def tools_to_gemini_json(tools: list) -> list[dict]:
            gemini_tools = []
            for t in tools:
                schema = getattr(t.args_schema, 'schema', lambda: {})()
                schema_flat = flatten_schema(schema)
                gemini_tools.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema_flat
                })
            return gemini_tools



        gemini_ready_tools = tools_to_gemini_json(tools)
        self.tools = gemini_ready_tools
    
        # print("[CUSTOM LLM]--------------------------------")
        # print(json.dumps(gemini_ready_tools, indent=2))
        return self

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }