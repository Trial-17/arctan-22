import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

def log_gemini_call(source: str, input_data: Dict[str, Any], output_data: Any = None, error: str = None):
    """
    Log un appel Gemini dans un fichier JSON horodaté à la racine du projet.
    
    Args:
        source: Identifiant de la source ("agent.py" ou "custom_llm.py")
        input_data: Dictionnaire contenant les paramètres d'entrée
        output_data: Résultat de l'appel (optionnel si erreur)
        error: Message d'erreur (optionnel)
    """
    try:
        # Trouver la racine du projet (arctan-22)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        
        # Créer le nom du fichier avec horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Millisecondes
        log_filename = f"gemini_logs_{timestamp}.json"
        log_path = project_root / log_filename
        
        # Préparer les données de log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "input": input_data,
            "output": output_data,
            "error": error,
            "success": error is None
        }
        
        # Écrire dans le fichier
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        # Ne pas bloquer l'exécution si le logging échoue
        print(f"⚠️ Erreur lors du logging Gemini: {str(e)}")

