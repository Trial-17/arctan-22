
import librosa
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from pathlib import Path

def create_app_structure():
    """
    Crée la structure de dossiers de l'application dans Documents/Adobe/Premiere Pro/Premiere Copilot/
    Retourne un dictionnaire contenant les chemins de tous les dossiers créés.
    """
    # Chemin de base
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot" / "temp"

    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    return base_path


def analyser_et_visualiser_silence_amélioré(
    input_path: str,
    db_threshold: float,
    min_silence_duration: float,
    min_segment_duration: float,
    padding: float,
    offset: float = 0.0,
    preview: bool = True
) -> List[List[float]]:
    """
    Analyse le silence dans un fichier audio.
    Si preview=True, analyse 15s à partir de l'offset et génère une image.
    Si preview=False, analyse tout le fichier sans générer d'image.
    """
    try:
        load_offset = offset if preview else 0.0
        duration = 15.0 if preview else None
        y, sr = librosa.load(input_path, sr=None, mono=True, offset=load_offset, duration=duration)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier audio : {e}")
        return [], None

    # Conversion des durées en échantillons
    min_silence_samples = int(min_silence_duration * sr)
    min_segment_samples = int(min_segment_duration * sr)
    padding_samples = int(padding * sr)

    # Détection des segments non-silencieux

    frame_length = 4096
    # frame_length = 3072
    hop_length = 2048
    # hop_length = 1536

    non_silent_intervals = librosa.effects.split(
        y, top_db=db_threshold, frame_length=frame_length, hop_length=hop_length
    )

    # Filtrage et application du padding
    kept_segments = []
    if non_silent_intervals.size > 0:
        for start, end in non_silent_intervals:
            if end - start > min_segment_samples:
                padded_start = max(0, start - padding_samples)
                padded_end = min(len(y), end + padding_samples)
                kept_segments.append([padded_start, padded_end])

    if not kept_segments:
        cuts_in_samples = [[0, len(y)]]
    else:
        # Génération de la liste des silences à couper
        cuts_in_samples = []
        last_end = 0
        for start, end in kept_segments:
            if start > last_end:
                silence_duration = (start - last_end) / sr
                if silence_duration >= min_silence_duration:
                    cuts_in_samples.append([last_end, start])
            last_end = end
        
        if last_end < len(y):
            silence_duration = (len(y) - last_end) / sr
            if silence_duration >= min_silence_duration:
                cuts_in_samples.append([last_end, len(y)])
    
    # Ajout de l'offset pour replacer les temps dans le contexte global du fichier
    cuts_in_seconds = [[round(s / sr + load_offset, 4), round(e / sr + load_offset, 4)] for s, e in cuts_in_samples]

    if not preview:
        return cuts_in_seconds, None

    # Génération du graphe pour le mode preview
    silence_mask = np.zeros_like(y, dtype=bool)
    for start_sample, end_sample in cuts_in_samples:
        silence_mask[start_sample:end_sample] = True

    y_sounding = y.copy()
    y_sounding[silence_mask] = np.nan
    y_silent = y.copy()
    y_silent[~silence_mask] = np.nan
    
    DPI = 100
    FIG_WIDTH_INCHES = 300 / DPI * 10
    FIG_HEIGHT_INCHES = 65 / DPI * 10
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), dpi=DPI)
    
    time_axis = np.linspace(0, len(y) / sr, num=len(y))
    ax.fill_between(time_axis, y_sounding, color="#006EFF", step=None)
    ax.fill_between(time_axis, y_silent, color="red", step=None)
    
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    base_path = create_app_structure()
    final_output_path = str(base_path / "waveform_silence.png")
    
    fig.savefig(final_output_path, transparent=True, dpi=DPI, pad_inches=0)
    plt.close(fig)
    print(cuts_in_seconds)
    return cuts_in_seconds, final_output_path