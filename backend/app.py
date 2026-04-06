import sys
import os
import time
import json
import traceback
import tempfile
import logging
import platform
import datetime
from logging.handlers import RotatingFileHandler

# --- Dépendances forcées pour le bundle PyInstaller ---
# Ces imports garantissent que PyInstaller embarque ces librairies
# sans avoir besoin d'utiliser des dizaines de --hidden-import.
try:
    import requests
    import zipfile
    import io
    import runpy
    from cryptography.fernet import Fernet

    import numpy as np
    import pandas as pd
    import librosa
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    
    import fastapi
    import uvicorn
    from starlette.responses import StreamingResponse
    import asyncio
    
    from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage, HumanMessage
    from langchain.tools import tool
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    
    import av
    from PIL import Image, ImageFont, ImageDraw, ImageFilter
    import websocket
    import srt
    import wave
    import pydantic
    
    import threading
    import signal
    import uuid
    import ast
    import inspect
    import struct
    import subprocess
    import random
    import re
    import copy
    import string
    import base64
    import difflib
    import math
except Exception as e:
    # Si un import plante au démarrage pur, on le laisse passer ici
    # car l'erreur sera de toute façon levée plus bas et traitée par handle_crash
    pass
# --------------------------------------------------------

API_URL = "https://api.premierecopilot.com/api/snake"
SECRET_KEY = b'GePQj013G8efbA3u3iKlooYbDqrnPkXZpfxaYJo7jRM='
CRASH_REPORT_URL = "https://api.premierecopilot.com/api/crashes/report-crash"

# Détermination du dossier utilisateur local (Application Support / LOCALAPPDATA)
def get_app_data_dir():
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        return os.path.join(base, "PremiereCopilot")
    elif platform.system() == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "PremiereCopilot")
    else:
        return os.path.join(os.path.expanduser("~"), ".premierecopilot")

APP_DIR = get_app_data_dir()
LOGS_DIR = os.path.join(APP_DIR, "logs")

# Initialisation du logger
def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "app.log")
    
    logger = logging.getLogger("PremiereCopilot")
    logger.setLevel(logging.DEBUG)
    
    # On évite d'ajouter multiples handlers s'il est déjà setup
    if not logger.handlers:
        # Rotating File Handler: Max 5MB, 5 backups
        fh = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        
        # En production, on retire l'output standard des "vrais logs" pour ne pas polluer l'IPC JSONL du frontend,
        # Sauf si on veut explicitement débug. On va commenter le StreamHandler pour laisser stdout propre.
        # sh = logging.StreamHandler(sys.stdout)
        # sh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        # sh.setFormatter(formatter)
        
        logger.addHandler(fh)
        # logger.addHandler(sh)
    
    return logger, log_file

logger, LOG_FILE_PATH = setup_logging()

def get_recent_logs(lines=500):
    """Lecture des dernières lignes du fichier de logs pour le rapport d'erreur"""
    try:
        if not os.path.exists(LOG_FILE_PATH):
            return "No log file found."
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Could not read logs: {e}"

def handle_crash(exception, context="main"):
    """
    Gestion des crash fatals: 
    1. Loggue l'erreur
    2. Imprime le JSON IPC dans la sortie standard (pour le frontend)
    3. Upload au serveur
    4. Quitte le process avec Code 1
    """
    tb_data = traceback.format_exc()
    error_type = type(exception).__name__
    error_message = str(exception)
    
    # Ecriture dans le log
    logger.error(f"FATAL CRASH ({context}): {error_type} - {error_message}")
    logger.error(tb_data)
    
    # --- 1. IPC to Frontend via STDOUT ---
    # Le frontend Premiere Pro écoute la sortie standard. Ce JSON va l'avertir du crash instantanément.
    ipc_event = {
        "__copilot_event__": True,
        "status": "error",
        "context": context,
        "error_type": error_type,
        "message": error_message,
        "traceback": tb_data
    }
    print(json.dumps(ipc_event), flush=True)

    # --- 2. Upload to Global API ---
    payload = {
        "error": error_message,
        "error_type": error_type,
        "context": context,
        "traceback": tb_data,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": datetime.datetime.now().isoformat(),
        "logs": get_recent_logs(500)
    }
    
    try:
        requests.post(CRASH_REPORT_URL, json=payload, timeout=10)
        logger.info("Crash report successfully uploaded to server.")
    except Exception as upload_error:
        logger.error(f"Failed to upload crash report: {upload_error}")

    # --- 3. Exit propre mais avec erreur ---
    sys.exit(1)

def run_remote_app():
    temp_dir_obj = None
    try:
        logger.info("--- Starting PremiereCopilot Launcher ---")
        cipher = Fernet(SECRET_KEY)
        
        # 1. Download with Retry Logic
        max_retries = 3
        encrypted_data = None
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading bundle (attempt {attempt+1}/{max_retries})...")
                r = requests.get(API_URL, timeout=30)
                r.raise_for_status()
                encrypted_data = r.json().get("payload")
                if encrypted_data:
                    logger.info("Bundle downloaded successfully.")
                    break
            except Exception as e:
                logger.warning(f"Download attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise Exception(f"Failed to download bundle after {max_retries} attempts. Last error: {e}")

        # 2. Decryption
        logger.info("Decrypting bundle...")
        zip_bytes = cipher.decrypt(encrypted_data.encode())

        # 3. Extraction dans un dossier temporaire
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        logger.info(f"Extracting bundle to {temp_dir}...")
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for member in z.infolist():
                try:
                    # Ignore root/current dir entries that throw errors
                    if member.filename in ['.', './', '/', '']:
                        continue
                    z.extract(member, temp_dir)
                except FileExistsError:
                    # Safe to ignore, specific to Windows trying to recreate the existing temp_dir
                    pass
                except Exception as e:
                    logger.warning(f"Issue extracting {member.filename}: {e}")
            
        sys.path.insert(0, temp_dir)
        target_script = os.path.join(temp_dir, "api.py")
        
        if not os.path.exists(target_script):
            raise FileNotFoundError(f"api.py missing in bundle. Extracted files: {os.listdir(temp_dir)}")

        # 4. Execution Bloquante
        logger.info(f"Executing dynamic script module: {target_script}...")
        
        # runpy va exécuter l'application et bloquer le thread tant qu'Uvicorn tourne
        runpy.run_path(target_script, run_name="__main__")

    except Exception as e:
        # Catch 100% des erreurs (download foiré, décryptage, ou l'api qui crashe elle-même en plein run)
        handle_crash(e, context="runtime_execution")
        
    finally:
        # Nettoyage si existant lors de l'extinction
        if temp_dir_obj:
            try:
                temp_dir_obj.cleanup()
                logger.info("Temporary directory cleaned up.")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")
        logger.info("--- PremiereCopilot Launcher Shutdown ---")

if __name__ == "__main__":
    run_remote_app()