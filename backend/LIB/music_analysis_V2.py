
# from pathlib import Path

# def create_app_structure_music_analysis():
#     """
#     Crée la structure de dossiers de l'application.
#     """
#     base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
#     path = base_path / "music_analysis"
#     path.mkdir(parents=True, exist_ok=True)
#     return path

# def analyze_music(filepath):
#     import allin1
    
#     dir = create_app_structure_music_analysis()
#     result = allin1.analyze(filepath, keep_byproducts=True, out_dir=dir, demix_dir = dir)
#     file_stem = Path(filepath).stem
#     json_path = dir / f"{file_stem}.json"
#     demixed_path = dir / f"htdemucs/{file_stem}"
#     return result.downbeats, result.beats, result.segments, demixed_path, json_path 


import subprocess
from pathlib import Path
import librosa
import json
import numpy as np
# import os
# import torch
# import torchaudio
# from demucs.apply import apply_model
# from demucs.pretrained import get_model
# from demucs.audio import save_audio

def create_app_structure_music_analysis():
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    path = base_path / "music_analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path

def separate_audio_with_demucs(audio_path, output_dir=None, model="htdemucs"):
    audio_path = str(Path(audio_path).resolve())
    output_dir = Path(output_dir) if output_dir else Path.cwd() / "demucs_output"
    output_dir = output_dir.resolve()

    cmd = [
        "demucs",
        "-n", model,
        "-o", str(output_dir),
        audio_path
    ]
    
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)

    stem_folder = output_dir / model / Path(audio_path).stem
    print(f"✅ Séparation terminée. Fichiers disponibles dans : {stem_folder}")
    return stem_folder


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

def analyze_music(filepath, modele_choice):
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

    # Séparer en stems avec Demucs
    print(modele_choice)
    if modele_choice == "PRO": 
        try:
            stem_folder = separate_audio_with_demucs(filepath, output_dir=dir_base )
        except Exception as e:
            print(f"Erreur lors de la séparation audio : {e}")
            stem_folder = ""    
    else: 
        stem_folder = ""
    return downbeats.tolist(), beat_times.tolist(), str(stem_folder), str(json_path)

