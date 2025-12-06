import asyncio

import shutil
from enum import Enum
from pydantic import BaseModel, Field
from pathlib import Path
import json
from typing import TypedDict, Annotated, List, Dict, Any
import operator
import time
import uuid
import requests
import base64

import os
import re

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



def configure_export_preset(params):
    """
    Configure and update the export preset with specified parameters
    
    Args:
        params: Export parameters dictionary containing:
            - presetName (str, optional): Name for the preset (default: "CUSTOM_EXPORT_PRESET")
            - width (int): Video width in pixels (e.g., 3840 for 4K, 1920 for FHD)
            - height (int): Video height in pixels (e.g., 2160 for 4K, 1080 for FHD)
            - fps (float): Framerate (e.g., 25, 30, 60)
            - targetBitrate (float): Target bitrate in Mbps (e.g., 80)
            - maxBitrate (float): Maximum bitrate in Mbps (e.g., 96)
            - doAudio (bool, optional): Enable audio export (default: True)
            - audioBitrate (int, optional): Audio bitrate in kbps (default: 320)
            - bitrateMode (str, optional): "CBR", "VBR_1PASS", or "VBR_2PASS" (default: "CBR")
            - useMaxQuality (bool, optional): Enable "Use Maximum Render Quality" (default: False)
        preset_template_path (str, optional): Path to the preset template file
            (default: "/Users/louisbolzinger/Documents/Projets/AutoPodcast_APP/EXPORT_PRESET.epr")
    
    Returns:
        Dict containing:
            - success (bool): Whether the operation succeeded
            - description (str): Description of the result
            - presetPath (str or None): Path to the modified preset file
    """
    try:
        # Define preset template path
        documents_preset_dir = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
        documents_preset_path = documents_preset_dir / "EXPORT_PRESET.epr"
        documents_preset_dir.mkdir(parents=True, exist_ok=True)
        
        if not documents_preset_path.exists():
            project_root = Path(__file__).parent.parent.parent.parent
            source_preset = project_root / "render" / "EXPORT_PRESET.epr"
            if source_preset.exists():
                shutil.copy2(source_preset, documents_preset_path)
                print(f"  📦 Preset copied to: {documents_preset_path}")

        preset_template_path = documents_preset_path
        
        # Validate input params
        if not isinstance(params, dict):
            return {
                "success": False,
                "description": "Invalid parameters: params must be a dictionary.",
                "presetPath": None
            }
        
        # Check if preset file exists
        preset_file = Path(preset_template_path)
        if not preset_file.exists():
            return {
                "success": False,
                "description": f"Preset template file not found at: {preset_template_path}",
                "presetPath": None
            }
        
        # Read the preset file
        with open(preset_file, 'r', encoding='utf-8') as f:
            preset_content = f.read()
        
        # Helper: Calculate FPS tick value (fps * 254016000000)
        def calculate_fps_ticks(fps):
            return round(254016000000 / fps)
        
        # Helper: Map bitrate mode to value
        def get_bitrate_encoding_value(mode):
            mode_map = {
                "CBR": "0",
                "VBR_1PASS": "2",
                "VBR_2PASS": "3"
            }
            return mode_map.get(mode, "0")  # Default to CBR
        
        # Apply parameters with defaults
        preset_name = params.get("presetName", "CUSTOM_EXPORT_PRESET")
        width = params.get("width", 3840)
        height = params.get("height", 2160)
        fps = params.get("fps", 25)
        target_bitrate = params.get("targetBitrate", 80)
        max_bitrate = params.get("maxBitrate", 96)
        do_audio = params.get("doAudio", True)
        audio_bitrate = params.get("audioBitrate", 320)
        bitrate_mode = params.get("bitrateMode", "CBR")
        use_max_quality = params.get("useMaxQuality", False)
        
        fps_ticks = calculate_fps_ticks(fps)
        bitrate_encoding_value = get_bitrate_encoding_value(bitrate_mode)
        
        # 1. Update Preset Name
        preset_content = re.sub(
            r'<PresetName>[^<]*</PresetName>',
            f'<PresetName>{preset_name}</PresetName>',
            preset_content
        )
        
        # 2. Update DoAudio
        preset_content = re.sub(
            r'<DoAudio>[^<]*</DoAudio>',
            f'<DoAudio>{"true" if do_audio else "false"}</DoAudio>',
            preset_content
        )
        
        # 3. Handle UseMaximumRenderQuality in StandardFilters section
        if use_max_quality:
            # If true, add it if missing or update if present
            if '<UseMaximumRenderQuality>' not in preset_content:
                # Insert after <CropType>0</CropType>
                preset_content = re.sub(
                    r'(<CropType>0</CropType>)',
                    r'\1\n\t\t<UseMaximumRenderQuality>true</UseMaximumRenderQuality>',
                    preset_content
                )
            else:
                preset_content = re.sub(
                    r'<UseMaximumRenderQuality>[^<]*</UseMaximumRenderQuality>',
                    '<UseMaximumRenderQuality>true</UseMaximumRenderQuality>',
                    preset_content
                )
        else:
            # If false, remove the tag if it exists
            preset_content = re.sub(
                r'\s*<UseMaximumRenderQuality>[^<]*</UseMaximumRenderQuality>\s*',
                '',
                preset_content
            )
            # Clean up potential double newlines
            preset_content = re.sub(r'\n\n\n+', '\n\n', preset_content)
        
        # 4. Update Video Width (Reverse Search: Value is BEFORE Identifier)
        preset_content = re.sub(
            r'(<ParamValue>)[^<]*(</ParamValue>(?:(?!<ParamValue>|<ParamIdentifier>)[\s\S])*?<ParamIdentifier>ADBEVideoWidth</ParamIdentifier>)',
            rf'\g<1>{width}\g<2>',
            preset_content
        )

        # 5. Update Video Height (Reverse Search: Value is BEFORE Identifier)
        preset_content = re.sub(
            r'(<ParamValue>)[^<]*(</ParamValue>(?:(?!<ParamValue>|<ParamIdentifier>)[\s\S])*?<ParamIdentifier>ADBEVideoHeight</ParamIdentifier>)',
            rf'\g<1>{height}\g<2>',
            preset_content
        )

        # 6. Update FPS (Reverse Search: Value is BEFORE Identifier)
        preset_content = re.sub(
            r'(<ParamValue>)[^<]*(</ParamValue>(?:(?!<ParamValue>|<ParamIdentifier>)[\s\S])*?<ParamIdentifier>ADBEVideoFPS</ParamIdentifier>)',
            rf'\g<1>{fps_ticks}\g<2>',
            preset_content
        )
        
        # 7. Update Bitrate Mode (VBR/CBR)
        bitrate_mode_map = {"CBR": 0, "VBR_1PASS": 1, "VBR_2PASS": 2}
        bitrate_val = bitrate_mode_map.get(bitrate_mode, 1)
        
        preset_content = re.sub(
            r'(<ParamValue>)[^<]*(</ParamValue>(?:(?!<ParamValue>|<ParamIdentifier>)[\s\S])*?<ParamIdentifier>ADBEVideoBitrateEncoding</ParamIdentifier>)',
            rf'\g<1>{bitrate_val}\g<2>',
            preset_content
        )

        # 8. Update Target Bitrate (Mbps)
        preset_content = re.sub(
            r'(<ParamValue>)[^<]*(</ParamValue>(?:(?!<ParamValue>|<ParamIdentifier>)[\s\S])*?<ParamIdentifier>ADBEVideoTargetBitrate</ParamIdentifier>)',
            rf'\g<1>{target_bitrate}\g<2>',
            preset_content
        )

        # 9. Update Max Bitrate (Mbps)
        preset_content = re.sub(
            r'(<ParamValue>)[^<]*(</ParamValue>(?:(?!<ParamValue>|<ParamIdentifier>)[\s\S])*?<ParamIdentifier>ADBEVideoMaxBitrate</ParamIdentifier>)',
            rf'\g<1>{max_bitrate}\g<2>',
            preset_content
        )

        # 10. Update Audio Enabled
        preset_content = re.sub(
            r'(<DoAudio>)[^<]*(</DoAudio>)',
            rf'\g<1>{str(do_audio).lower()}\g<2>',
            preset_content
        )

        # 11. Update Audio Bitrate (kbps)
        preset_content = re.sub(
            r'(<ParamValue>)[^<]*(</ParamValue>(?:(?!<ParamValue>|<ParamIdentifier>)[\s\S])*?<ParamIdentifier>ADBEAudioBitrate</ParamIdentifier>)',
            rf'\g<1>{audio_bitrate}\g<2>',
            preset_content
        )

        # 12. Update Maximum Render Quality
        if use_max_quality:
            preset_content = re.sub(
                r'(<ParamValue>)[^<]*(</ParamValue>(?:(?!<ParamValue>|<ParamIdentifier>)[\s\S])*?<ParamIdentifier>ADBEVideoUseMaxRenderQuality</ParamIdentifier>)',
                r'\g<1>true\g<2>',
                preset_content
            )
        else:
            preset_content = re.sub(
                r'(<ParamValue>)[^<]*(</ParamValue>(?:(?!<ParamValue>|<ParamIdentifier>)[\s\S])*?<ParamIdentifier>ADBEVideoUseMaxRenderQuality</ParamIdentifier>)',
                r'\g<1>false\g<2>',
                preset_content
            )
        
        # Write the updated preset back to the file
        with open(preset_file, 'w', encoding='utf-8') as f:
            f.write(preset_content)
        
        return {
            "success": True,
            "description": f"Export preset configured successfully: {preset_name} ({width}x{height} @ {fps}fps, {target_bitrate}Mbps, {bitrate_mode}, Audio: {'ON' if do_audio else 'OFF'}, MaxQuality: {'ON' if use_max_quality else 'OFF'})",
            "presetPath": str(preset_file)
        }
    
    except Exception as error:
        return {
            "success": False,
            "description": f"Error while configuring export preset: {str(error)}",
            "presetPath": None
        }


# ------------- EXPORT TOOL FOR AGENT -------------
class ExportSequence(BaseModel):
    prompt: str = Field(description="The user's request for export (e.g., 'export in 4k', 'export for instagram', 'lightweight export', 'just export it')")

@tool("export_sequence", args_schema=ExportSequence)
async def export_sequence(prompt: str):
    """
    Export the active sequence to the Documents folder using Adobe Media Encoder.
    
    This tool:
    1. Analyzes the active sequence settings (resolution, fps)
    2. Uses an LLM to determine the best export parameters based on your prompt and the sequence settings
    3. Configures the export preset
    4. Triggers the export in Adobe Media Encoder
    5. It will not start the rendering, but preapre it in Media Encoder
    """
    try:
        config.API_STATUS = "Reading sequence settings..."
        
        # Step 1: Get Sequence Settings from Premiere Pro
        call_id = str(uuid.uuid4())
        PENDING_JS_CALLS[call_id] = {
            "args": {"script": "$._MYFUNCTIONS.getSequenceSettings();"},
            "result": None,
            "status": "pending"
        }
        
        # Wait for JS result
        timeout = 15
        start_time = time.time()
        seq_settings = None
        
        while time.time() - start_time < timeout:
            if call_id in PENDING_JS_CALLS and PENDING_JS_CALLS[call_id]["status"] == "completed":
                result_str = PENDING_JS_CALLS[call_id]["result"]
                del PENDING_JS_CALLS[call_id]
                try:
                    seq_settings = json.loads(result_str)
                except:
                    return f"Error parsing sequence settings: {result_str}"
                break
            await asyncio.sleep(0.5)
            
        if call_id in PENDING_JS_CALLS:
            del PENDING_JS_CALLS[call_id]
            
        if not seq_settings or not seq_settings.get("success"):
            return "Error: Could not retrieve active sequence settings. Make sure a sequence is open and active."

        print(f"✅ Sequence found: {seq_settings.get('name')} ({seq_settings.get('width')}x{seq_settings.get('height')} @ {seq_settings.get('fps')}fps)")

        # Step 2: Ask LLM for Export Parameters
        config.API_STATUS = "Determining export parameters..."
        
        full_prompt = f"""
        ### ACTIVE SEQUENCE SETTINGS:
        - Name: {seq_settings.get('name')}
        - Resolution: {seq_settings.get('width')}x{seq_settings.get('height')}
        - Framerate: {seq_settings.get('fps')} fps
        
        ### USER REQUEST:
        {prompt}
        
        ### TASK:
        Determine the best export parameters based on the user request and the sequence settings.
        If the user doesn't specify resolution/fps, KEEP the sequence settings.
        If the user asks for specific format (Instagram, 4K, etc.), adapt the settings.
        """
        
        system_instruction = """
        You are an expert video engineer. Your task is to define the technical parameters for a video export.
        
        RULES:
        - Unless specified otherwise by the user, ALWAYS keep the source resolution and framerate.
        - For bitrates:
          - 4K (UHD): target ~60-80 Mbps
          - 1080p (FHD): target ~15-25 Mbps
          - 720p (HD): target ~5-10 Mbps
          - Social Media (Instagram/TikTok): target ~8-15 Mbps
        - WorkArea: 0=Entire Sequence, 1=In/Out Points. Default to 0 unless user mentions "selection" or "in out".
        """
        
        structured_output = {
            "type": "object",
            "properties": {
                "fileName": {"type": "string", "description": "Output filename (without extension). Use sequence name if not specified."},
                "width": {"type": "integer", "description": "Video width"},
                "height": {"type": "integer", "description": "Video height"},
                "fps": {"type": "number", "description": "Framerate"},
                "targetBitrate": {"type": "number", "description": "Target bitrate in Mbps"},
                "maxBitrate": {"type": "number", "description": "Max bitrate in Mbps (usually target + 20%)"},
                "workArea": {"type": "integer", "description": "0=Entire, 1=In/Out, 2=WorkArea"},
                "bitrateMode": {"type": "string", "enum": ["CBR", "VBR_1PASS", "VBR_2PASS"]},
                "useMaxQuality": {"type": "boolean", "description": "Render at maximum quality"}
            },
            "required": ["fileName", "width", "height", "fps", "targetBitrate", "maxBitrate", "workArea"]
        }
        
        # Call LLM
        export_params = gemini_call(full_prompt, system_instruction, structured_output, None, 0.2, "gemini-2.5-flash-lite")
        
        if not export_params:
            return "Error: Failed to determine export parameters."
            
        print(f"✅ Export Params: {json.dumps(export_params, indent=2)}")

        # Step 3: Configure Preset
        config.API_STATUS = "Configuring export preset..."
        
        preset_config = {
            "presetName": f"EXPORT_{export_params['fileName']}",
            "width": export_params['width'],
            "height": export_params['height'],
            "fps": export_params['fps'],
            "targetBitrate": export_params['targetBitrate'],
            "maxBitrate": export_params['maxBitrate'],
            "doAudio": True,
            "audioBitrate": 320,
            "bitrateMode": export_params.get('bitrateMode', 'VBR_1PASS'),
            "useMaxQuality": export_params.get('useMaxQuality', False)
        }
        
        preset_result = configure_export_preset(preset_config)
        
        if not preset_result["success"]:
            return f"Error configuring preset: {preset_result['description']}"
            
        # Step 4: Trigger Export
        config.API_STATUS = "Triggering export..."
        
        call_id = str(uuid.uuid4())
        js_args = json.dumps({
            "fileName": export_params['fileName'],
            "workArea": export_params['workArea'], 
            "presetPath": preset_result['presetPath']
        })
        
        PENDING_JS_CALLS[call_id] = {
            "args": {"script": f"$._MYFUNCTIONS.exportSequenceToDocuments({js_args});"},
            "result": None,
            "status": "pending"
        }
        
        # Wait for export trigger
        timeout = 120
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
            return "Error: Timed out waiting for export to start."
            
        # Parse result
        try:
            export_result = json.loads(result)
            if export_result.get("success"):
                return f"""✅ Export lauched!
                
            **File:** {export_result.get('outputPath')}
            **Settings:** {export_params['width']}x{export_params['height']} @ {export_params['fps']}fps ({export_params['targetBitrate']} Mbps)
            **Job ID:** {export_result.get('jobId')}

            The file has been sent to Adobe Media Encoder. The default export path is your Documents folder."""
            else:
                return f"❌ Export failed: {export_result.get('description')}"
        except:
            return f"Export result: {result}"

    except Exception as e:
        return f"Error in export_sequence: {str(e)}"
