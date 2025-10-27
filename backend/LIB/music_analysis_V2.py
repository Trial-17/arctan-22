import subprocess
from pathlib import Path
import librosa
import json
import numpy as np


def create_app_structure_music_analysis():
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    path = base_path / "music_analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_beats_and_downbeats(filename):
    y, sr = librosa.load(filename)
    tempo, beats_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Convertir les frames des beats en secondes
    beats_times = librosa.frames_to_time(beats_frames, sr=sr)

    # Calculer l'énergie RMS
    rms = librosa.feature.rms(y=y)[0]
    # Normaliser les valeurs d'énergie pour la comparaison
    rms = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)


    # Étape 4 (simplifiée) : Combiner les deux informations
    beat_scores = []
    tolerance_window = 0.05  # 50 ms

    for beat_time in beats_times:
        # Trouver l'index de temps RMS le plus proche du beat
        closest_rms_idx = np.argmin(np.abs(rms_times - beat_time))
        
        # Chercher la valeur RMS maximale dans la fenêtre de tolérance
        start_idx = np.argmin(np.abs(rms_times - (beat_time - tolerance_window)))
        end_idx = np.argmin(np.abs(rms_times - (beat_time + tolerance_window)))
        
        score = np.max(rms[start_idx:end_idx+1])
        beat_scores.append(score)

    beat_scores = np.array(beat_scores)

    # Étape 5 (simplifiée) : Sélectionner les downbeats (en supposant du 4/4)
    downbeats = []
    # On part de l'hypothèse que le premier temps est un candidat potentiel
    # et on regarde tous les 4 temps lequel est le plus fort en moyenne
    best_phase = -1
    max_avg_score = -1

    # On suppose une mesure de 4 temps
    period = 4
    for i in range(period):
        # Calcule la moyenne des scores pour cette "phase" (0, 1, 2, or 3)
        phase_scores = beat_scores[i::period]
        avg_score = np.mean(phase_scores)
        
        if avg_score > max_avg_score:
            max_avg_score = avg_score
            best_phase = i

    # Les downbeats sont les beats à la meilleure phase
    downbeat_indices = np.arange(best_phase, len(beats_times), period)
    downbeats_times = beats_times[downbeat_indices]
    
    return beats_times, downbeats_times

def analyze_music(filepath):
    dir_base = create_app_structure_music_analysis()
    filename = Path(filepath).stem
    dir_output = dir_base / filename
    dir_output.mkdir(parents=True, exist_ok=True)

    # Analyse audio avec Librosa        
    beat_times , downbeats = get_beats_and_downbeats(filepath)
    
    json_data = {

        "beats":  beat_times.tolist(),
        "downbeats": downbeats.tolist(),
    }
    json_path = dir_output / "beat_data.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return downbeats.tolist(), beat_times.tolist(), str(json_path)

