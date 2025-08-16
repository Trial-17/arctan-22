import json
import requests
import copy
import string
from typing import List, Dict, Any
import os
import srt
from datetime import timedelta
from datetime import datetime
from pathlib import Path
import warnings
import requests
from LIB import config
import base64
import re
warnings.filterwarnings("ignore")


silence_threshold_G =  3
max_chars_srt, tolerance_srt, nb_lignes_srt = 40, 20, 2

def create_app_structure_transcription_analysis():
    """
    Crée la structure de dossiers de l'application.
    """
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    path = base_path / "transcription_analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_parameters(prompt, token):


    prompt = f"""
    Tu es un monteur vidéo professionnel. Tu dois m’aider à fluidifier un discours brut.
    Le texte suivant est le **prompt user** ou il indique sont intention de montage, sotn contexte, et éventuellement des paramètres

    **Ton objectif :** Remplir les **paramètres de sous titrage** d'après l'intention du user. La liste des paramètres est la suivante: 
    - **max_chars_srt** : maximum de charactère par ligne de sous-titre : compris entre 20 et 150, plus proche de 20 pour un format Réseau Sociaux, et plus proche de 150 pour un format Youtube
    - **tolerance_srt** : tolérance de charactères pour le découpage des sous-titres : généralement entre 10 et 50, plus proche de 10 pour un format Réseau Sociaux, et plus proche de 50 pour un format Youtube
    - **nb_lignes_srt** : nombre de lignes maximum de sous-titre : généralement entre 1 et 3, 2 pour un format Réseau Sociaux, et 1 pour un format Youtube
    


    ### Prompt user à analyser :
    \"\"\"{prompt}\"\"\"


        """.strip()

    try:
        response = requests.post(
            f"{config.API_URL}/global-gemini-call",
            json={"prompt": prompt,
                  "model": "gemini-2.0-flash-lite",
                  "schema_type" : "speaker_parameters"
                  },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Erreur lors de l'amélioration du prompt: {response.status_code}")
            return {'max_chars_srt': 60, 'nb_lignes_srt': 3, 'tolerance_srt': 10}
            
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API d'amélioration de prompt: {str(e)}")
        return {'max_chars_srt': 60, 'nb_lignes_srt': 3, 'tolerance_srt': 10}

def suggest_cuts_from_transcript(transcript, token, modele_choice):

    prompt = f"""
    You are a professional video editor. You need to streamline a speech.

    The following text is a raw transcription, with errors, repetitions, hesitations, and reformulations.

    Your objective:
        •	Analayze and Identify the parts to remove that make the speech confusing such as reformulations, filler words, hesitations, or repetitions.
        •	Return the ranges of IDs corresponding to the sections to delete, can be full sentence, one word, few words, ...
        •	In case of a reformulation or repetition, try to keep the last version of the sentence, unless the user has specified otherwise.
        

    ### Transcription to analyse :
    \"\"\"{transcript}\"\"\"
        """.strip()
    modele = "gemini-2.5-pro-preview-05-06" if modele_choice == "PRO" else "gemini-2.5-flash-preview-05-20"
    try:
        response = requests.post(
            f"{config.API_URL}/global-gemini-call",
            json={"prompt": prompt,
                  "model": modele,
                  "schema_type" : "cuts_speakers"
                  },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Erreur lors de l'amélioration du prompt: {response.status_code}")
            return []
            
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API d'amélioration de prompt: {str(e)}")
        return []

def get_cut_times(words: list[dict], threshold: float, print_paragraph: bool = False) -> list[tuple[float, float]]:
    """
    Prend en entrée une liste de dictionnaires représentant des mots avec leurs temps,
    et retourne une liste d'intervalles (tuple avec temps_cut_start, temps_cut_end) à couper.
    
    Le premier intervalle va de 0 jusqu'au début du premier mot si la durée est supérieure ou égale au seuil.
    Ensuite, pour chaque mot suivant, si le gap entre la fin du mot précédent et le début
    du mot courant est supérieur ou égal au seuil, on considère ce gap comme un intervalle à couper.
    
    Si print_paragraph est True, la fonction reconstruit et affiche le paragraphe complet.
    
    :param words: Liste de dictionnaires contenant au moins les clés 'start' et 'end' et 'word'
    :param threshold: Durée minimale d'un intervalle pour qu'il soit considéré comme un cut
    :param print_paragraph: Si True, affiche le paragraphe reconstruit à partir des 'word'
    :return: Liste de tuples (temps_cut_start, temps_cut_end)
    """
    if not words:
        return []
    
    # Reconstituer le paragraphe en concaténant les 'word'
    paragraph = "".join(word["word"] for word in words)
    if print_paragraph:
        print("Paragraphe :", paragraph)
    
    cuts = []
    
    # Ajouter le premier intervalle : de 0 jusqu'au début du premier mot (s'il y a un silence suffisant)
    first_start = float(words[0]['start'])
    if first_start >= threshold:
        cuts.append((0.0, first_start))
    
    # Examiner les gaps entre chaque mot successif
    for i in range(1, len(words)):
        prev_end = float(words[i - 1]['end'])
        current_start = float(words[i]['start'])
        gap = current_start - prev_end
        if gap >= threshold:
            cuts.append((prev_end, current_start))
    
    return cuts

def segment_transcription(transcription, max_words_per_segment=30):
    """
    Transforme une transcription mot-à-mot en segments regroupés par phrases,
    puis fusionne les phrases en segments de taille limitée.

    Args:
        transcription (list): Liste de mots avec 'word', 'start', 'end'.
        max_words_per_segment (int): Nombre maximum de mots par segment.

    Returns:
        list: Liste de segments, chaque segment est une liste de mots.
    """
    phrases = []
    current_phrase = []

    # Étape 1 : découpage en phrases
    for word in transcription:
        current_phrase.append(word)
        if re.search(r"[.!?]$", word["word"]):  # Fin de phrase détectée
            phrases.append(current_phrase)
            current_phrase = []

    if current_phrase:  # Ajouter la dernière phrase si elle ne finit pas par ponctuation
        phrases.append(current_phrase)

    # Étape 2 : regroupement en chunks (segments)
    chunks = []
    current_chunk = []

    for phrase in phrases:
        total_words = len(current_chunk) + len(phrase)

        if not current_chunk:  # Premier ajout
            current_chunk.extend(phrase)

        elif total_words <= max_words_per_segment:
            current_chunk.extend(phrase)

        else:
            chunks.append({"segments": current_chunk})
            current_chunk = phrase  # Repartir avec la phrase courante

    if current_chunk:
        chunks.append({"segments": current_chunk})

    return chunks

def merge_transcript_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fusionne dans la liste `words` :
      - tout mot dont text.strip() commence par "'" avec son prédécesseur
      - tout mot dont text.strip() se termine par un signe de ponctuation
    """
    merged: List[Dict[str, Any]] = []
    punctuation = set(string.punctuation)
    
    for w in words:
        text = w['word']
        stripped = text.strip()
        
        # Condition de fusion : apostrophe en début, ou ponctuation en fin
        if merged and (
            stripped.startswith("'") or 
            (stripped and stripped[-1] in punctuation)
        ):
            prev = merged.pop()
            # Concaténation du texte
            new_word = prev['word'] + w['word']
            # Début = celui du mot précédent, fin = celui du mot courant
            new_start = prev['start']
            new_end   = w['end']
            # Moyenne des probabilités
            new_prob = (float(prev.get('probability', 1.0)) + float(w.get('probability', 1.0))) / 2
            
            merged.append({
                'word': new_word,
                'start': new_start,
                'end': new_end,
                'probability': new_prob
            })
        else:
            # Pas de fusion : on réplique tel quel
            merged.append({
                'word': w['word'],
                'start': w['start'],
                'end': w['end'],
                'probability': w.get('probability', 1.0)
            })
    
    return merged

def split_by_sentence_end(words: List[Dict]) -> List[List[Dict]]:
    """
    Découpe la transcription mot par mot en une liste de phrases,
    chaque phrase étant une liste de mots. La séparation se fait
    lorsqu'un mot se termine par '.', '?', ou '!'
    """
    sentence_endings = {'.', '?', '!'}
    sentences = []
    current_sentence = []

    for word in words:
        current_sentence.append(word)
        text = word["word"].strip()
        if text and text[-1] in sentence_endings:
            sentences.append(current_sentence)
            current_sentence = []

    # Ajouter la dernière phrase si elle ne s'est pas terminée par une ponctuation
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

def split_phrase_into_blocks(
    words: List[Dict[str, Any]],
    max_chars: int = 50,
    tolerance: int = 20
) -> List[Dict[str, Any]]:
    """
    Version intelligente qui découpe une phrase (liste de mots) en blocs :
    - Respecte max_chars + tolerance
    - Cherche à terminer les blocs sur une ponctuation naturelle (virgule, point, etc.)
    - Ne commence jamais un bloc par une ponctuation
    """
    blocks = []
    i = 0
    n = len(words)
    punct_end = {'.', '!', '?', ','}

    while i < n:
        block = []
        length = 0
        j = i
        best_end_j = None  # index idéal où s'arrêter (ponctuation rencontrée)

        while j < n:
            word = words[j]
            word_stripped = word['word'].strip()
            tentative_length = length + len(word_stripped)

            if length > 0 and tentative_length > (max_chars + tolerance):
                break

            block.append(word)
            length = tentative_length

            if word_stripped and word_stripped[-1] in punct_end:
                best_end_j = j + 1  # on inclura ce mot dans le bloc

            j += 1

            # Stop si on a atteint la taille et trouvé une ponctuation
            if length >= max_chars and best_end_j is not None:
                break

        # Choix du bon point de découpe
        if best_end_j is not None:
            block = words[i:best_end_j]
            j = best_end_j
        else:
            # Pas trouvé de ponctuation, on coupe à j
            block = words[i:j]

        # Évite de commencer un bloc par une ponctuation
        while block and block[0]['word'].strip() and block[0]['word'].strip()[-1] in punct_end:
            # On déplace la ponctuation au bloc précédent
            if blocks:
                prev = blocks[-1]
                prev['words'].append(block.pop(0))
                prev['word'] += prev['words'][-1]['word']
                prev['end'] = prev['words'][-1]['end']
            else:
                break  # pas de bloc précédent

        if not block:
            i += 1
            continue

        block_text = "".join(w['word'] for w in block).strip()
        block_start = block[0]['start']
        block_end = block[-1]['end']
        block_prob = sum(float(w.get('probability', 1.0)) for w in block) / len(block)

        blocks.append({
            'start': block_start,
            'end': block_end,
            'word': block_text,
            'probability': block_prob,
            'words': block  # utile pour debug
        })

        i = j

    return blocks

def seconds_to_timedelta(seconds: float) -> timedelta:
    return timedelta(seconds=seconds)

def phrases_to_srt(
    phrases: List[List[Dict]],
    max_chars: int = 40,
    tolerance: int = 10,
    nb_lignes: int = 2
) -> str:
    """
    Génère un fichier SRT à partir de phrases découpées en blocs, 
    où chaque sous-titre contient jusqu'à nb_lignes blocs de texte.
    Chaque nouvelle phrase commence un nouveau sous-titre.
    """
    subtitles = []
    idx = 1

    for phrase in phrases:
        blocks = split_phrase_into_blocks(phrase, max_chars=max_chars, tolerance=tolerance)
        
        # Regrouper les blocs en paquets de nb_lignes max
        for i in range(0, len(blocks), nb_lignes):
            chunk = blocks[i:i+nb_lignes]
            if not chunk:
                continue
            # Texte combiné (chaque bloc devient une ligne)
            lines = [blk['word'] for blk in chunk]
            full_text = "\n".join(lines)
            start = seconds_to_timedelta(chunk[0]['start'])
            end   = seconds_to_timedelta(chunk[-1]['end'])
            subtitles.append(srt.Subtitle(index=idx, start=start, end=end, content=full_text))
            idx += 1

    return srt.compose(subtitles)

def groc_transcription(audio_path, token):
    
    def map_words_to_segments(segments, words):
        """
        Associe à chaque segment la liste des mots qui le composent.

        Args:
            segments (list): Liste des segments (avec start / end).
            words (list): Liste des mots (avec start / end).

        Returns:
            list: Les segments enrichis avec une nouvelle clé 'words'.
        """
        for segment in segments:
            segment_start = segment["start"]
            segment_end = segment["end"]
            
            # Filtre les mots qui commencent dans la fenêtre temporelle du segment
            # avec une tolérance de 0.05s pour la fin
            segment_words = [
                word for word in words
                if segment_start <= word["start"] < segment_end and word["end"] <= segment_end 
            ]
            segment["words"] = segment_words
        return segments

    def clean_transcription(transcription_groc):
        for key in ["task", "duration", "x_groq", "words"]:
            transcription_groc.pop(key, None)  # évite une erreur si la clé n'existe pas
        return transcription_groc


        import base64

    def encode_audio_base64(audio_path):
        with open(audio_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded


    audio_base64 = encode_audio_base64(audio_path)
    
    # Prépare la requête à l'API
    filename = os.path.basename(audio_path)
    url = f"{config.API_URL}/speaker_analysis"  # Ajustez l'URL selon votre configuration
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"  # Utilisation du token pour l'authentification
    }
    
    payload = {
        "audio_base64": audio_base64,
        "filename": filename
    }
    
    # Appel à l'API
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Lève une exception si la requête a échoué
    
    # Parse la réponse
    result = response.json()
    transcription = result["transcription"]

        
    transcription_groc = copy.deepcopy(transcription)
    # segments = transcription_groc["segments"]
    # words = transcription_groc["words"]

    # segments_with_words = map_words_to_segments(segments, words)

    # transcription_groc["segments"] = segments_with_words

    # transcription_groc = clean_transcription(transcription_groc)


    return transcription_groc["words"]
    
def analyze_speaker(prompt, audio_path, token, modele_choice):


    config.API_STATUS = "Setting parameters"
    parameters = get_parameters(prompt, token)
    max_chars_srt = parameters["max_chars_srt"]
    tolerance_srt = parameters["tolerance_srt"]
    nb_lignes_srt = parameters["nb_lignes_srt"]
    
    config.API_STATUS = "Transcribing audio"

    transcription = groc_transcription(audio_path, token)

    # nettoyage de la transcription
    transcription_corrected = copy.deepcopy(transcription)

    
    config.API_STATUS = "Fluidifying script"
    chunks = segment_transcription(transcription_corrected, max_words_per_segment=50)

    k = 0
    cuts_index = []
    words_from_whisper_FULL = []
    for z, chunk in enumerate(chunks):

        words_from_whisper_ID = []
        words_from_whisper = []
        # print(chunk, z)
        words = chunk['segments']

        for i in range(len(words)):
            words_from_whisper_FULL.append(words[i])
            words_from_whisper_ID.append(( k , words[i]['word']))
            k += 1
                

        cuts = suggest_cuts_from_transcript(words_from_whisper_ID, token, modele_choice)['cuts']

        
    
        for i in range (len(cuts)):
            cuts_index.append((cuts[i]['id_start'], cuts[i]['id_end']))
        

    indices = []
    for start, end in cuts_index:
        indices.extend(range(start, end + 1)) 
        
    filtered_words = [word for i, word in enumerate(words_from_whisper_FULL) if i not in indices]
    filtered_words = [{'word': ' ' + w['word'], 'start': w['start'], 'end': w['end']} for w in filtered_words]

    cuts_to_premiere = get_cut_times(filtered_words, silence_threshold_G, False)
    
    
    cuts_to_premiere_saved = copy.deepcopy(cuts_to_premiere)
    filtered_words_saved = copy.deepcopy(filtered_words)

    duration_list = []
    duration_total = 0
    for i in range (len(cuts_to_premiere_saved)):
        duration = cuts_to_premiere_saved[i][1] - cuts_to_premiere_saved[i][0]
        duration_total += duration
        duration_list.append(duration)

    for i in range(len(filtered_words_saved)): 
        for k in range (len(cuts_to_premiere)):
            if filtered_words[i]['start'] >= cuts_to_premiere[k][1] :
                filtered_words_saved[i]['start'] = filtered_words_saved[i]['start'] - duration_list[k]
                filtered_words_saved[i]['end'] = filtered_words_saved[i]['end'] - duration_list[k]

    # for i in range(1, len(filtered_words_saved)):
    #     if filtered_words_saved[i]['start'] < filtered_words_saved[i-1]['end']:
    #         print('Error --- overlapping words')
    #         print(filtered_words_saved[i]['start'] - filtered_words_saved[i-1]['end'])

    config.API_STATUS = "Exporting smart SRT"
    filtered_words_saved_1 = merge_transcript_words(filtered_words_saved)

    filtered_words_saved_1 = merge_transcript_words(filtered_words_saved)
    phrases = split_by_sentence_end(filtered_words_saved_1)


    cuts_to_premiere_safe = []
    deltas = []
    shift = 0.5
    total_shift = 0
    total_temps = 0
    for i in range (len(cuts_to_premiere)):
        if cuts_to_premiere[i][0] > 0.1:
            cuts_to_premiere_safe.append((cuts_to_premiere[i][0] + shift, cuts_to_premiere[i][1]))

        else:
            cuts_to_premiere_safe.append((cuts_to_premiere[i][0], cuts_to_premiere[i][1]))
            
        if i < len(cuts_to_premiere) - 1:
                total_shift += shift
                total_temps += cuts_to_premiere[i +1][0] - cuts_to_premiere[i][1]
                deltas.append((total_temps, total_shift))
        
    phrases_shifted = copy.deepcopy(phrases)
    for phrase in phrases_shifted:
        for word in phrase:
            for i in range (len(deltas)-1):
                if word['start'] >= deltas[i][0] and word['end'] < deltas[i+1][0]:
                    word['start'] = word['start'] + deltas[i][1]
                    word['end'] = word['end'] + deltas[i][1]
                elif word['start'] >= deltas[-1][0]:

                    word['start'] = word['start'] + deltas[-1][1]
                    word['end'] = word['end'] + deltas[-1][1]

                    break

    srt_output = phrases_to_srt(phrases_shifted, max_chars_srt, tolerance_srt, nb_lignes_srt)

    def create_app_structure_transcription_analysis():
        """
        Crée la structure de dossiers de l'application.
        """
        base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
        path = base_path / "transcription_analysis"
        path.mkdir(parents=True, exist_ok=True)
        return path

    dir = create_app_structure_transcription_analysis()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path_SRT = dir / f"result_for_premiere_{timestamp}.srt"

    with open(export_path_SRT, "w", encoding="utf-8") as f:
        f.write(srt_output)

    srt_output_analyze = phrases_to_srt(phrases_shifted, max_chars=50, tolerance=20, nb_lignes=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path_SRT_A = dir / f"result_for_analyse_{timestamp}.srt"


    with open(export_path_SRT_A, "w", encoding="utf-8") as f:
        f.write(srt_output_analyze)


    # print(export_path_SRT, export_path_SRT_A, json.dumps(cuts_to_premiere_safe))
    
    return export_path_SRT, export_path_SRT_A, cuts_to_premiere_safe