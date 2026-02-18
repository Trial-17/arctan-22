import requests
import zipfile
import io
import sys
import tempfile
import runpy
from cryptography.fernet import Fernet

import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


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



from PIL import Image, ImageFont, ImageDraw

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
    SECRET_KEY = b'GePQj013G8efbA3u3iKlooYbDqrnPkXZpfxaYJo7jRM='
    cipher = Fernet(SECRET_KEY)

    try:
        # 1. Download
        print("Downloading bundle...")
        r = requests.get(API_URL)
        r.raise_for_status()
        encrypted_data = r.json().get("payload")

        # 2. Decryption
        zip_bytes = cipher.decrypt(encrypted_data.encode())

        # 3. Extraction dans un dossier temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Extraction in {temp_dir}...")
            
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                z.extractall(temp_dir)
            
            # 4. Configuration de l'environnement
            # Important : On ajoute le dossier temporaire au sys.path
            # pour que api.py puisse faire "import agent" ou "import config"
            sys.path.insert(0, temp_dir)
            
            # Cible : le fichier api.py extrait
            target_script = os.path.join(temp_dir, "api.py")
            
            if not os.path.exists(target_script):
                print(f"Error : api.py not found in the zip !")
                # Debug : afficher ce qu'il y a dans le zip
                print("Files received :", os.listdir(temp_dir))
                return

            # 5. Execution "Main"
            print(f"Launching {target_script}...")
            try:
                # run_name="__main__" force l'exécution du bloc if __name__ == "__main__"
                runpy.run_path(target_script, run_name="__main__")
            except Exception as e:
                print(f"Error during script execution : {e}")

    except Exception as e:
        print(f"Fatal error : {e}")

if __name__ == "__main__":
    run_remote_app()