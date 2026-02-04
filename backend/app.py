import requests
import zipfile
import io
import sys
import tempfile
import runpy
from cryptography.fernet import Fernet

import os
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')


from pathlib import Path
from typing import List, Optional, Dict, Any, TypedDict, Annotated, Iterator


import threading
import signal
import uuid
import asyncio
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from fastapi import FastAPI, HTTPException, Request
import uvicorn

import json
import logging
import traceback
import shutil
from enum import Enum
from pydantic import BaseModel, Field
import operator


from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage, HumanMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import av
import platform


from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import StructuredTool



from PIL import Image

import ast
import inspect
from datetime import datetime
from datetime import timedelta
import datetime
import wave
import struct
import subprocess
import librosa
import random
import re
import pandas as pd
import numpy as np
import copy
import string
import srt
import base64
import copy
import difflib
import math











def run_remote_app():
    API_URL = "https://api.premierecopilot.com/api/snake"
    VERSION_URL = "https://api.premierecopilot.com/api/snake/version"
    SECRET_KEY = b'GePQj013G8efbA3u3iKlooYbDqrnPkXZpfxaYJo7jRM='
    cipher = Fernet(SECRET_KEY)
    
    # Dossier de persistance pour l'application téléchargée
    # user_data_dir = os.path.join(os.path.expanduser("~"), ".premieregpt_backend")
    # Pour l'instant, on reste en local ou temp si on préfère, mais pour le cache il vaut mieux un dossier fixe.
    # On va utiliser un dossier temporaire persistent pour cette session.
    # Ou mieux : le dossier courant/backend_live
    live_dir = os.path.join(os.getcwd(), "backend_live")
    os.makedirs(live_dir, exist_ok=True)

    current_process = None
    current_hash = None
    
    print(f"🚀 Launcher started. Monitoring updates from {VERSION_URL}")

    while True:
        try:
            # 1. Check Version (Remote)
            try:
                r_ver = requests.get(VERSION_URL, timeout=10)
                if r_ver.status_code == 200:
                    remote_hash = r_ver.json().get("version")
                else:
                    print(f"⚠️ API Info check failed: {r_ver.status_code}")
                    remote_hash = None
            except Exception as e:
                print(f"⚠️ Network error checking version: {e}")
                remote_hash = None
            
            # 2. Update logic
            update_needed = False
            if remote_hash and remote_hash != current_hash:
                print(f"🔄 New version detected: {remote_hash} (Was: {current_hash})")
                update_needed = True
            
            # Si on n'a jamais téléchargé (démarrage), on force l'update
            if current_hash is None and not os.path.exists(os.path.join(live_dir, "api.py")):
                update_needed = True

            if update_needed:
                print("⬇️ Downloading new bundle...")
                try:
                    r = requests.get(API_URL, timeout=60)
                    r.raise_for_status()
                    encrypted_data = r.json().get("payload")
                    zip_bytes = cipher.decrypt(encrypted_data.encode())
                    
                    # Stop processus précédent
                    if current_process:
                        print("🛑 Stopping previous process...")
                        # Sur Windows, kill() est assez violent, terminate() est mieux
                        current_process.terminate()
                        try:
                            current_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            current_process.kill()
                        current_process = None

                    # Nettoyage et Extraction
                    # On vide le dossier live proprement
                    for filename in os.listdir(live_dir):
                        file_path = os.path.join(live_dir, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print(f'Failed to delete {file_path}. Reason: {e}')

                    # Extraction
                    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                        z.extractall(live_dir)
                    
                    print(f"✅ Extracted to {live_dir}")
                    current_hash = remote_hash

                    # Démarrage
                    target_script = "api.py" # Relatif au cwd qui sera live_dir
                    print(f"▶️ Starting {target_script}...")
                    
                    # On lance dans un subprocess indépendant
                    # On passe sys.executable pour utiliser le même python (et ses libs installées)
                    env = os.environ.copy()
                    env["PYTHONPATH"] = live_dir + os.pathsep + env.get("PYTHONPATH", "")
                    
                    current_process = subprocess.Popen(
                        [sys.executable, target_script],
                        cwd=live_dir,
                        env=env
                    )
                    
                except Exception as e:
                    print(f"❌ Update failed: {e}")
                    # On réessaiera au prochain tour de boucle
            
            # 3. Monitor Process
            if current_process:
                ret = current_process.poll()
                if ret is not None:
                    print(f"⚠️ Child process exited with code {ret}. Restarting in 5s...")
                    current_process = None
                    current_hash = None # Force re-download/re-check on next loop logic potentially
                    # Mais si c'est juste un crash, on veut peut-être juste relancer sans re-télécharger si le hash n'a pas changé.
                    # Pour simplifier : on ne reset PAS current_hash, le script relancera au prochain tour si update_needed est False mais qu'on veut assurer la présence.
                    # Ajout d'une logique de restart simple:
                    time.sleep(5)
                    # La boucle va reprendre, si hash == remote_hash, update_needed=False.
                    # Il nous faut un mécanisme pour relancer si crashé.
                    # On le fait ici simplement :
                    if current_hash is not None and os.path.exists(os.path.join(live_dir, "api.py")):
                         print(f"♻️ Reviving process...")
                         env = os.environ.copy()
                         env["PYTHONPATH"] = live_dir + os.pathsep + env.get("PYTHONPATH", "")
                         current_process = subprocess.Popen(
                            [sys.executable, "api.py"],
                            cwd=live_dir,
                            env=env
                        )

            # Wait before next check
            time.sleep(30) # Vérifie toutes les 30 secondes

        except KeyboardInterrupt:
            print("\n👋 Stopping Launcher...")
            if current_process:
                current_process.terminate()
            break
        except Exception as e:
            print(f"Fatal Launcher Error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    run_remote_app()