import random
import librosa
import numpy as np
import re
from LIB import config
import pandas as pd
import requests

def transform_front_to_modal(audio_files, front_data):
    modalFormData = {}
    
    # Speakers
    speaker_mapping = {}
    for speaker in front_data['speakers']:
        # audioTrack est une string "1", "2", etc. donc int() -1 pour indexer audio_files
        speaker_mapping[speaker['name']] = int(speaker['audioTrack'])

    # Cameras
    for camera in front_data['cameras']:
        video_track = camera['videoTrack']
        frequency = camera['frequency'].capitalize()
        
        # Nettoyer les noms de speakers
        selected_speakers = [
            sp['name'].replace('\n', '').replace('×', '').strip()
            for sp in camera['selectedSpeakers']
        ]
        modalFormData[video_track] = ", ".join(selected_speakers)
        modalFormData[f"{video_track} Freq"] = frequency
    
    # Audio Tracks
    for idx, audio_path in enumerate(audio_files):
        audio_track_name = f"Audio Track {idx+1}"
        # Trouver quel speaker est associé à cette piste
        linked_speaker = None
        for speaker, track_idx in speaker_mapping.items():
            if track_idx == idx+1:
                linked_speaker = speaker
                break
        if linked_speaker:
            modalFormData[audio_track_name] = linked_speaker

    # Cut frequency remplacé par Cut Min et Cut Max
    modalFormData["Cut Min"] = str(front_data['global']['minCutTime'])
    modalFormData["Cut Max"] = str(front_data['global']['maxCutTime'])
    
    return modalFormData, audio_files

def transform_video_modal_data(modal_data):
    # Dictionnaires temporaires pour stocker les informations
    tracks = {}
    audio_tracks = {}
    cut_min = None
    cut_max = None

    # Patterns pour détecter les clés vidéo, audio et leur fréquence
    video_key_pattern = re.compile(r"VideoTrack (\d+)$", re.IGNORECASE)
    pace_key_pattern  = re.compile(r"VideoTrack (\d+) Freq$", re.IGNORECASE)
    audio_key_pattern = re.compile(r"Audio Track (\d+)$", re.IGNORECASE)
    speaker_pattern   = re.compile(r"Speaker\s*(\d+)", re.IGNORECASE)

    # Parcourir les clés du formulaire pour identifier les Video et Audio Tracks
    for key, value in modal_data.items():
        m_video = video_key_pattern.match(key)
        m_audio = audio_key_pattern.match(key)

        if m_video:
            track_num = int(m_video.group(1))
            speakers = [int(m_sp.group(1)) - 1 for s in value.split(",") if (m_sp := speaker_pattern.search(s.strip()))]
            if track_num not in tracks:
                tracks[track_num] = {"speaker": set(speakers)}
            else:
                tracks[track_num]["speaker"].update(speakers)

        elif m_audio:
            track_num = int(m_audio.group(1))
            speakers = [int(m_sp.group(1)) - 1 for s in value.split(",") if (m_sp := speaker_pattern.search(s.strip()))]
            if track_num not in audio_tracks:
                audio_tracks[track_num] = {"speaker": set(speakers)}
            else:
                audio_tracks[track_num]["speaker"].update(speakers)

        elif key.lower() == "cut min":
            cut_min = int(value.strip())
        
        elif key.lower() == "cut max":
            cut_max = int(value.strip())

    # Associer les fréquences aux Video Tracks
    for key, value in modal_data.items():
        m_pace = pace_key_pattern.match(key)
        if m_pace:
            track_num = int(m_pace.group(1))
            if track_num not in tracks:
                tracks[track_num] = {"speaker": set(), "frequence": value.strip()}
            else:
                tracks[track_num]["frequence"] = value.strip()

    # Construire la liste des caméras
    camera_list = [{
        "track": new_index,
        "speaker": sorted(list(tracks[track_num].get("speaker", set()))),
        "frequence": tracks[track_num].get("frequence")
    } for new_index, track_num in enumerate(sorted(tracks.keys()))]

    # Construire la liste des pistes audio
    audio_list = [{
        "track": new_index,
        "speaker": sorted(list(audio_tracks[track_num].get("speaker", set())))
    } for new_index, track_num in enumerate(sorted(audio_tracks.keys()))]

    return camera_list, audio_list, cut_min, cut_max

def gap_overlapped_by_other(gap, other_intervals):
    """
    Vérifie si le gap (défini par (gap_start, gap_end)) est recouvert,
    en tout ou en partie, par l'un quelconque des intervalles de l'autre speaker.
    """
    gap_start, gap_end = gap
    for o_start, o_end in other_intervals:
        # S'il y a un recouvrement : l'intervalle de l'autre speaker intersecte le gap
        if o_start < gap_end and o_end > gap_start:
            return True
    return False

def detecter_temps_de_parole_multi(file_paths, audio_list, lissage_duree_sec=5.0, frame_length=4096, hop_length=2048):
    """
    Analyse de 1 à N fichiers audio, détecte les segments de parole pour chaque locuteur
    et retourne un dictionnaire avec les timestamps.

    Args:
        file_paths (list): Liste des chemins vers les fichiers audio.
        audio_list (list): Description des pistes et des locuteurs.
        lissage_duree_sec (float): Durée en secondes pour la fenêtre de lissage.

    Returns:
        dict: Un dictionnaire où les clés sont les ID des locuteurs et les valeurs
              sont des listes de tuples (start_time, end_time).
    """
    num_tracks = len(file_paths)
    if num_tracks == 0 or num_tracks != len(audio_list):
        raise ValueError("Le nombre de 'file_paths' doit correspondre au nombre d''audio_list' et être supérieur à 0.")

    all_rms = []
    speaker_ids = []
    sr = 0 # On récupèrera le sample rate du premier fichier
    max_time = 0
    # 1. Traitement audio pour chaque piste
    # On charge et on calcule le RMS pour tous les fichiers
    for i in range(num_tracks):
        audio, current_sr = librosa.load(file_paths[i], sr=None)
        if i == 0:
            sr = current_sr
            # Récupérer la durée de l'audio en secondes pour la première piste
            max_time = librosa.get_duration(y=audio, sr=sr)


        
        # Calculer l'énergie RMS (puissance)
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        all_rms.append(rms)
        speaker_ids.append(audio_list[i]['speaker'][0])

    times = librosa.times_like(all_rms[0], sr=sr, hop_length=hop_length)

    # Cas particulier : s'il n'y a qu'une seule piste, elle parle tout le temps
    if num_tracks == 1:
        return {speaker_ids[0]: [(times[0], times[-1])]}, max_time

    # 2. Lissage généralisé
    # Lisser les valeurs RMS de toutes les pistes
    hop_duration_sec = hop_length / sr
    window_size_frames = int(lissage_duree_sec / hop_duration_sec)
    
    # On met les RMS dans un DataFrame pour un lissage facile
    df_rms = pd.DataFrame(np.array(all_rms).T) # Transposé pour avoir les pistes en colonnes
    
    all_rms_smoothed = []
    for col in df_rms.columns:
        smoothed_rms = df_rms[col].rolling(window=window_size_frames, min_periods=1, center=True).mean().to_numpy()
        all_rms_smoothed.append(smoothed_rms)
    
    # 3. Détection du locuteur dominant (la plus grosse modification)
    # On convertit toutes les pistes lissées en dB
    all_db_final = [librosa.amplitude_to_db(rms, ref=np.max) for rms in all_rms_smoothed]
    
    # On empile les listes de dB dans une matrice (pistes x temps)
    db_matrix = np.vstack(all_db_final)
    
    # np.argmax(..., axis=0) trouve l'INDEX de la piste la plus forte pour chaque instant
    dominant_track_indices = np.argmax(db_matrix, axis=0)
    
    # On mappe ces indices aux vrais ID des locuteurs
    speaker_ids_array = np.array(speaker_ids)
    locuteur_actif_par_trame = speaker_ids_array[dominant_track_indices]

    # 4. Algorithme de segmentation (cette partie n'a pas besoin de changer !)
    segments = {}
    current_speaker = None
    segment_start_time = 0

    for i, speaker_id in enumerate(locuteur_actif_par_trame):
        if current_speaker is None:
            current_speaker = speaker_id
            segment_start_time = times[i]

        elif speaker_id != current_speaker:
            segment_end_time = times[i]
            segments.setdefault(current_speaker, []).append((segment_start_time, segment_end_time))
            
            current_speaker = speaker_id
            segment_start_time = times[i]
    
    if current_speaker is not None:
        final_end_time = times[-1]
        segments.setdefault(current_speaker, []).append((segment_start_time, final_end_time))

    return segments, max_time

def merge_time(seg_dict):
    """
    Pour chaque speaker dans seg_dict (par exemple, {0: [...], 1: [...]}),
    fusionne deux intervalles consécutifs si le gap qui les sépare n'est pas
    recouvert par un intervalle de l'autre speaker.
    
    Paramètres :
      - seg_dict (dict): Dictionnaire avec pour clé l'ID du speaker et
                         pour valeur une liste d'intervalles (tuples (start, end)).
    
    Retourne :
      - Un nouveau dictionnaire avec les intervalles fusionnés pour chaque speaker.
    """
    merged = {}
    speakers = sorted(seg_dict.keys())
    
    # Pour chaque speaker, on définit l'autre speaker (ici on suppose 2 speakers)
    for speaker in speakers:
        # Si plus d'un speaker, on définit l'autre comme la réunion de tous les autres
        # (ici pour 2 speakers, c'est simplement l'autre)
        other_speakers = []
        for s in speakers:
            if s != speaker:
                other_speakers.extend(seg_dict.get(s, []))
        # On trie les intervalles du speaker courant par temps de début
        intervals = sorted(seg_dict.get(speaker, []), key=lambda x: x[0])
        if not intervals:
            merged[speaker] = []
            continue
        
        merged_intervals = []
        current_start, current_end = intervals[0]
        
        for next_interval in intervals[1:]:
            next_start, next_end = next_interval
            gap = (current_end, next_start)
            # Si le gap n'est pas recouvert par un intervalle de l'autre speaker, on fusionne.
            if not gap_overlapped_by_other(gap, other_speakers):
                # Fusionner : on étend la fin de l'intervalle courant.
                current_end = next_end
            else:
                # Sinon, on sauvegarde l'intervalle courant et on démarre un nouveau.
                merged_intervals.append((current_start, current_end))
                current_start, current_end = next_interval
        
        merged_intervals.append((current_start, current_end))
        merged[speaker] = merged_intervals
    
    return merged

def add_start_time(segments):
    """
    Ajuste les intervalles afin que le temps de départ le plus bas soit ramené à 0.
    
    Paramètres :
      - segments (dict) : Dictionnaire sous la forme {speaker: [(start, end), ...]}
    
    Retourne :
      - Un nouveau dictionnaire avec tous les intervalles ajustés.
    """
    # Recherche du plus petit temps de départ dans tous les segments
    min_time = min(s for speaker in segments for (s, e) in segments[speaker])
    normalized = {}
    for speaker in segments:
        normalized[speaker] = [(s - min_time, e - min_time) for (s, e) in segments[speaker]]
    return normalized

def get_timeline(speaker_times):
    flattened = []
    for speaker_id in range(len(speaker_times)):
        
        for k in speaker_times[speaker_id]:
            start, end = k
            flattened.append({
                "speaker": speaker_id,
                "start": start
            })

    
    flattened.sort(key=lambda x: x['start'])
    return flattened

def apply_min_time_timeline(timeline, min_gap=1.0):
    smoothed = []
    smoothed.append(timeline[0])

    for i in range(1, len(timeline)-1):
        entry = timeline[i]
        next_entry = timeline[i+1]
        
        gap =  next_entry['time'] - entry['time']
        # if gap < min_gap and next_entry['speaker'] == last_entry['speaker']:
        if gap < min_gap:
            continue
        elif smoothed[-1]['camera'] == entry['camera']:
            continue
        else:
            smoothed.append(entry)
            
    smoothed.append(timeline[-1])
    return smoothed

def generate_cuts(flattened_starts, cut_min, cut_max, max_time):
    cuts = []
    
    
    if len(flattened_starts) == 1:
        # Cas d'un seul speaker
        current = flattened_starts[0]
        speaker = current['speaker']
        current_time = current['start']


        cuts.append({"speaker": speaker, "start": current_time})

        while True:
            jump = random.uniform(cut_min, cut_max)
            next_cut = current_time + jump

            if next_cut + cut_min > max_time +30:
                break

            cuts.append({"speaker": speaker, "start": next_cut})
            current_time = next_cut

        return cuts
    
    for i in range(len(flattened_starts) - 1):
        current = flattened_starts[i]
        next_start = flattened_starts[i + 1]['start']

        speaker = current['speaker']
        current_time = current['start']

        # On commence par le start actuel
        cuts.append({"speaker": speaker, "start": current_time})

        while True:
            # Générer un temps d'attente entre cut_min et cut_max
            jump = random.uniform(cut_min, cut_max)
            next_cut = current_time + jump

            # Si le prochain cut dépasse ou est trop proche du next_start, on arrête
            if next_cut + cut_min > next_start:
                break

            cuts.append({"speaker": speaker, "start": next_cut})
            current_time = next_cut

    # Ajouter le tout dernier point (optionnel selon besoin)
    last = flattened_starts[-1]
    cuts.append({"speaker": last['speaker'], "start": last['start']})

    return cuts

def assign_cameras_to_cuts(cuts, camera_list, freq_weights):
    assigned_cuts = []
    previous_camera = None

    for idx, cut in enumerate(cuts):
        speaker = cut['speaker']

        # Étape 1: filtrer les caméras avec ce speaker
        eligible_cameras = [
            cam for cam in camera_list if speaker in cam['speaker']
        ]

        # Si pas de caméra trouvée → toutes les caméras
        if not eligible_cameras:
            eligible_cameras = camera_list.copy()

        # Étape 2: retirer la caméra actuellement utilisée
        filtered_cameras = [
            cam for cam in eligible_cameras if cam['track'] != previous_camera
        ]

        # Si la liste est vide après retrait, reprendre toutes sauf celle utilisée
        if not filtered_cameras:
            filtered_cameras = [
                cam for cam in camera_list if cam['track'] != previous_camera
            ]

        # Si toujours vide (ex: 1 seule caméra existante), reprendre tout
        if not filtered_cameras:
            filtered_cameras = camera_list.copy()

        # Étape 3: normaliser les poids
        weights = [freq_weights[cam['frequence']] for cam in filtered_cameras]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Étape 4: tirage au sort
        selected_camera = random.choices(
            population=[cam['track'] for cam in filtered_cameras],
            weights=normalized_weights,
            k=1
        )[0]

        # Assigner
        assigned_cuts.append({
            "speaker": speaker,
            "start": cut['start'],
            "camera": selected_camera
        })
        # print(f" Speaker {speaker}, Camera {selected_camera}, FIltered Cameras: {filtered_cameras} ")
        previous_camera = selected_camera

    # POST-TRAITEMENT
    final_cuts = []
    for cut in assigned_cuts:
        final_cuts.append({
            "time": round(cut['start'], 3),
            "camera": cut['camera'], 
            "active_speaker": cut['speaker']
        })

    return final_cuts

def shift_timeline(timeline, shift):
    for i in range(1, len(timeline)):
        timeline[i]['time'] += shift
    return timeline

def separate_cameras(camera_list):

    
    list_speaker = []
    for camera in camera_list:
        list_speaker.extend(camera['speaker'])
    
    number_of_speaker = max(list_speaker) + 1
    list_speaker = list(range(number_of_speaker))
        
    camera_speaker_list = []
    camera_wide_list = []
    for camera in camera_list:
        if len(camera['speaker']) == 1:
            camera_speaker_list.append(camera)
        else:
            camera_wide_list.append(camera)
       
    # trier camera wide list par nbr de speaker 
    camera_wide_list.sort(key=lambda x: len(x['speaker']), reverse=True)     
    
    
    # vérifier que chaque speaker a au moins une caméra
    for speaker in list_speaker:
        if not any(speaker in camera['speaker'] for camera in camera_speaker_list):
            # Si le speaker n'a pas de caméra dédiée, on lui assigne la première camera wide ou il figure
            for camera in camera_wide_list:
                if speaker in camera['speaker']:
                    camera_speaker_list.append(camera)
                    break
            
    return camera_speaker_list, camera_wide_list
    
def assign_wide_cameras(result, camera_wide_list, percentage_map , freq_weights,threshold ):
    """
    Analyse une timeline pour placer intelligemment des plans larges
    en fonction des enchaînements de coupes et des caméras disponibles.
    """
    
    # === Étape 0 : Construction des chaines ===

    # 1. On initialise la nouvelle clé dans chaque dictionnaire de 'result'
    for item in result:
        item['chain_count'] = 0

    # 2. Le calcul reste identique, mais on modifie 'result' directement
    i = 0
    n = len(result)
    while i < n - 1:
        chain_links = 0
        j = i
        
        while j < n - 1 and (result[j+1]['time'] - result[j]['time']) < threshold:
            chain_links += 1
            j += 1
        
        # Si une chaîne est trouvée, on met à jour le dictionnaire au début de la chaîne
        if chain_links > 0:
            result[i]['chain_count'] = chain_links
        
        # On saute les coupes déjà comptées
        if chain_links > 0:
            i += chain_links
        else:
            i += 1
    
    # === Étape 1 : Nettoyage initial ===
    # On ignore les chaînes de 1, qui ne sont pas de vrais enchaînements.
    for item in result:
        if item.get('chain_count', 0) == 1:
            item['chain_count'] = 0
    
    # === Étape 2 : Déterminer la fréquence globale ===
    freq_order = ["Very low", "Low", "Medium", "High"]
    available_freqs = [cam['frequence'] for cam in camera_wide_list]
    global_freq = "Very low"
    for freq in reversed(freq_order):
        if freq in available_freqs:
            global_freq = freq
            break


    # === Étape 3 : Identifier et trier les emplacements potentiels ===
    potential_locations = []
    for i, item in enumerate(result):
        if item.get('chain_count', 0) > 0:
            potential_locations.append({
                'index': i,
                'time': item['time'],
                'chain_count': item['chain_count']
            })

    # Tri : d'abord par 'chain_count' décroissant, puis aléatoirement
    potential_locations.sort(key=lambda x: (x['chain_count'], random.random()), reverse=True)

    # === Étape 4 : Sélectionner un pourcentage des emplacements ===

    percentage_to_keep = percentage_map.get(global_freq, 0.2)
    
    num_to_keep = int(len(potential_locations) * percentage_to_keep)
    selected_locations = potential_locations[:num_to_keep]


    # === Étape 5 : Traiter chaque emplacement sélectionné ===


    for location in selected_locations:
        start_index = location['index']
        # La chaîne d'événements va de l'index de départ jusqu'à l'index + chain_count
        end_index = start_index + location['chain_count']

        # 5.1. Trouver les intervenants actifs dans le segment
        speakers_in_segment = set()
        for i in range(start_index, end_index):
            speakers_in_segment.add(result[i]['active_speaker'])
        
        # 5.2. Filtrer les caméras larges possibles
        possible_cams = []
        for cam in camera_wide_list:
            # La caméra est valide si elle couvre TOUS les intervenants du segment
            if set(cam['speaker']).issuperset(speakers_in_segment):
                possible_cams.append(cam)

        # Fallback si aucune caméra ne couvre tout le monde
        if not possible_cams:
            best_score = 0
            fallback_cams = []
            for cam in camera_wide_list:
                score = len(set(cam['speaker']).intersection(speakers_in_segment))
                if score > best_score:
                    best_score = score
                    fallback_cams = [cam]
                elif score == best_score and best_score > 0:
                    fallback_cams.append(cam)
            possible_cams = fallback_cams

        # Si aucune caméra ne correspond (même en fallback), on saute ce segment
        if not possible_cams:
            print(f"Aucune caméra large trouvée pour le segment à {location['time']}s. On ignore.")
            continue

        # 5.3. Tirage au sort pondéré de la caméra
        cam_weights = [freq_weights[cam['frequence']] for cam in possible_cams]
        chosen_camera = random.choices(possible_cams, weights=cam_weights, k=1)[0]
        
        # 5.4. Appliquer le changement de caméra dans 'result'
        # Le plan large couvre toute la durée de la chaîne de coupes rapides
        # print(f"Placement du plan large (cam {chosen_camera['track']}) de {result[start_index]['time']}s à {result[end_index + 1]['time'] if end_index + 1 < len(result) else 'la fin'}.")
        for i in range(start_index, end_index):
            result[i]['camera'] = chosen_camera['track']
    
    # === Étape 6 : Nettoyage final pour enlever les coupes redondantes ===
    if not result:
        return []

    cleaned_result = [result[0]]
    for i in range(1, len(result)):
        # On ajoute la coupe suivante seulement si la caméra est différente de la précédente
        if result[i]['camera'] != cleaned_result[-1]['camera']:
            cleaned_result.append(result[i])

    return cleaned_result

def main_podcast(file_paths, front_data, user_token=None):
    
    freq_weights = {"High": 0.5, "Medium": 0.3, "Low": 0.15, "Very low": 0.05}
    percentage_map = {"High": 0.8, "Medium": 0.6, "Low": 0.4, "Very low": 0.2}
    threshold = 1.5
    
    lissage_duree_sec = 0.80
    ignoreCutLessThan = front_data['global']['ignoreCutLessThan']
    delayCuts = front_data['global']['delayCuts']
    
    
    # print(file_paths)
    # print(front_data)


    if user_token is not None:
        try:
            headers = {"Authorization": f"Bearer {user_token}"}
            access_url = f"{config.API_URL}/user/podcast-access"
            resp = requests.get(access_url, headers=headers)
            resp.raise_for_status()
            access_data = resp.json()
            if not access_data.get("podcast_access", False):
                raise PermissionError("L'utilisateur n'a pas accès au podcast (abonnement insuffisant).")
        except Exception as e:
            raise PermissionError(f"Erreur lors de la vérification d'accès podcast: {e}")
    


    modalFormData, file_paths = transform_front_to_modal(file_paths, front_data)
    camera_list, audio_list, cut_min, cut_max = transform_video_modal_data(modalFormData)

    camera_speaker_list, camera_wide_list = separate_cameras(camera_list)
    
    speaker_times, max_time = detecter_temps_de_parole_multi(file_paths, audio_list, lissage_duree_sec) 

    # plot_audio_levels_multi(file_paths, audio_list, lissage_duree_sec) 
    speaker_times = merge_time(speaker_times)
    speaker_times = add_start_time(speaker_times)
    
    timeline_V0= get_timeline(speaker_times)

    # 
    if cut_max != 0:
        timeline_V1 = generate_cuts(timeline_V0, cut_min, cut_max, max_time) # application du max cut

        timeline_V2 = assign_cameras_to_cuts(timeline_V1, camera_list, freq_weights) # application des caméras speaker

    # print(camera_speaker_list, camera_wide_list)
    else:
        timeline_V2 = assign_cameras_to_cuts(timeline_V0, camera_speaker_list, freq_weights) # application des caméras speaker
    timeline_V3 = assign_wide_cameras(timeline_V2, camera_wide_list, percentage_map, freq_weights, threshold) # application des caméras wide

    timeline_V4 = apply_min_time_timeline(timeline_V3, min_gap=ignoreCutLessThan) # application du ignore cut less than 
    
    timeline_V5 = shift_timeline(timeline_V4, delayCuts) # application du delay cuts

    return timeline_V5

