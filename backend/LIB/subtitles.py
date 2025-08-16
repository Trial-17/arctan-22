import json
import base64
import json
import copy
import requests
import difflib
import math
from pathlib import Path
import datetime
import os
from LIB import config



def Set_Majuscule(transcription_phrases):
    transcription_phrases[0][0]['word'] = transcription_phrases[0][0]['word'].capitalize()
    for phrase, i in zip(transcription_phrases[:-1], range(len(transcription_phrases)-1)):
        if phrase[-1]['word'][-1] in [".", "!", "?"]:
            transcription_phrases[i+1][0]['word'] = transcription_phrases[i+1][0]['word'].capitalize()

    return transcription_phrases

def Get_Segments(transcription):

    transcription_phrases = []
    phrase = []

    for word, i in zip(transcription['words'], range(len(transcription['words']))):
        end = word['end']
        next_start = transcription['words'][i + 1]['start'] if i + 1 < len(transcription['words']) else None
        phrase.append(word)
        if word['word'][-1] in [".", "!", "?", ";", ":", ","]:
            transcription_phrases.append(phrase)
            phrase = []
        elif next_start > end : 
            transcription_phrases.append(phrase)
            phrase = []

    return transcription_phrases

def _afficher_phrase(phrase, preview=False):
    result = ""
    for word in phrase:
        result += f"{word['word']} "
    if preview:
        print(result.strip())
    return result.strip()

def Spelling_Correction(segment_text, user_prompt, model, token):
    
    try:
        response = requests.post(
            f"{config.API_URL}/spelling-correction",
            json={"prompt": user_prompt, "transcription": str(segment_text), "model": str(model)},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Erreur lors de l'appel à l'API spelling correction: {response.status_code}")

            
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API spelling correction: {str(e)}")
  
def synchroniser_corrections(batch_original, segments_corriges):
    """
    Met à jour la structure 'batch_original' avec les corrections textuelles
    de 'segments_corriges'.

    - Applique les modifications de mots (remplacements).
    - Ignore les mots ajoutés par le LLM.
    - Gère correctement les positions pour modifier les bons mots.
    - Affiche les opérations en temps réel.
    """
    # print("--- Début de la synchronisation des corrections ---")

    # 1. Aplatir la structure 'batch' pour faciliter les manipulations
    # On garde une liste de tous les dictionnaires de mots dans l'ordre
    flat_batch = [mot_dict for segment in batch_original for mot_dict in segment]

    # Créer une liste de mots depuis le batch original pour la comparaison
    # On enlève la ponctuation pour une meilleure correspondance avec le texte du LLM
    mots_originaux = [d['word'].strip('.,!?') for d in flat_batch]

    # Créer une liste de mots à partir des segments corrigés par le LLM
    texte_corrige = " ".join(segments_corriges)
    mots_corriges = texte_corrige.split()

    # 2. Utiliser SequenceMatcher pour trouver les différences
    # autojunk=False est important pour ne pas ignorer de détails
    matcher = difflib.SequenceMatcher(None, mots_originaux, mots_corriges, autojunk=False)

    # 3. Parcourir les opérations et appliquer les modifications
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Un ou plusieurs mots ont été remplacés
            bloc_original = mots_originaux[i1:i2]
            bloc_corrige = mots_corriges[j1:j2]
            
            # On parcourt le bloc original pour appliquer les changements
            # On ne dépasse pas la longueur du bloc corrigé
            for k in range(len(bloc_original)):
                if k < len(bloc_corrige):
                    # L'index du mot à modifier dans notre liste 'flat_batch'
                    index_mot_a_modifier = i1 + k
                    mot_dict_original = flat_batch[index_mot_a_modifier]
                    nouveau_mot = mots_corriges[j1 + k]

                    # On vérifie si le mot a réellement changé pour ne pas polluer l'affichage
                    if mot_dict_original['word'] != nouveau_mot:
                        print(
                            f"✅ Correction : '{mot_dict_original['word']}' -> '{nouveau_mot}' "
                            f"(au temps {mot_dict_original['start']:.2f}s)"
                        )
                        # On applique la modification directement dans la structure de données
                        mot_dict_original['word'] = nouveau_mot
                        
        elif tag == 'insert':
            # Le LLM a ajouté des mots, on les ignore comme demandé
            mots_ajoutes = mots_corriges[j1:j2]
            # print(f"ℹ️ Ajout ignoré : Le LLM a inséré le(s) mot(s) \"{' '.join(mots_ajoutes)}\".")

    # print("\n--- Synchronisation terminée ---")
    
    # La structure originale 'batch_original' a été modifiée directement,
    # car les dictionnaires qu'elle contient sont mutables.
    return batch_original

def groc_transcription(audio_path, token, prompt=""):

    def encode_audio_base64(audio_path):
        with open(audio_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded

    audio_base64 = encode_audio_base64(audio_path)
    
    # Prépare la requête à l'API
    filename = os.path.basename(audio_path)
    url = f"{config.API_URL}/subtitles"  # Ajustez l'URL selon votre configuration
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"  # Utilisation du token pour l'authentification
    }
    
    payload = {
        "audio_base64": audio_base64,
        "filename": filename,
        "prompt": prompt
    }
    
    # Appel à l'API
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Lève une exception si la requête a échoué
    
    # Parse la réponse
    result = response.json()
    transcription = result["transcription"]

        
    transcription_groc = copy.deepcopy(transcription)


    return transcription_groc

def decouper_segment_intelligent(
    segment,
    max_caracteres,
    tolerance=10,
    penalite_depassement=50,
    penalite_mot_unique=150,
    penalite_diff_longueur=1,
    bonus_ponctuation=-50
):
    """
    Découpe un segment (liste de dictionnaires de mots) en blocs optimisés.
    Retourne une liste de blocs, où chaque bloc est une liste de dictionnaires.
    """
    if not segment:
        return []

    # Étape 1 : Extraire la liste des mots pour l'algorithme
    mots = [d['word'] for d in segment]
    n = len(mots)

    # Initialisation de la programmation dynamique
    scores = [0] + [math.inf] * n
    chemins = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(i):
            # --- Évaluation du bloc potentiel mots[j:i] ---
            bloc_mots_str = mots[j:i]
            longueur_bloc_actuel = len(" ".join(bloc_mots_str))
            
            # 1. Pénalité de longueur
            penalite = 0
            if longueur_bloc_actuel > max_caracteres + tolerance:
                continue # Découpage invalide
            elif longueur_bloc_actuel > max_caracteres:
                penalite += penalite_depassement * (longueur_bloc_actuel - max_caracteres)

            # 2. Pénalité pour mot unique (orphelin)
            if len(bloc_mots_str) == 1 and n > 1:
                penalite += penalite_mot_unique
            
            # 3. Bonus pour la ponctuation de fin de bloc
            if bloc_mots_str[-1].endswith((',', ';', ':')):
                penalite += bonus_ponctuation

            # 4. Pénalité pour la différence de longueur
            if j > 0:
                point_de_coupe_avant_j = chemins[j]
                longueur_bloc_precedent = len(" ".join(mots[point_de_coupe_avant_j:j]))
                diff_longueur = abs(longueur_bloc_actuel - longueur_bloc_precedent)
                penalite += diff_longueur * penalite_diff_longueur

            # --- Mise à jour des scores ---
            if scores[j] + penalite < scores[i]:
                scores[i] = scores[j] + penalite
                chemins[i] = j

    # --- Reconstruction du meilleur chemin EN CONSERVANT LA NOMENCLATURE ---
    blocs_de_dictionnaires = []
    index = n
    while index > 0:
        debut = chemins[index]
        # On tranche la liste de dictionnaires originale
        bloc_actuel = segment[debut:index]
        blocs_de_dictionnaires.insert(0, bloc_actuel)
        index = debut

    return blocs_de_dictionnaires

def ajuster_silences_inter_segments(segments, seuil_secondes):
    """
    Parcourt une liste de segments et fusionne les silences courts entre eux.

    Si la durée entre la fin d'un segment et le début du suivant est
    inférieure au seuil, la fin du premier segment est étendue pour
    combler le vide.

    Args:
        segments (list): La liste de listes de dictionnaires de mots.
        seuil_secondes (float): Le seuil en dessous duquel le silence est comblé.

    Returns:
        list: La liste de segments modifiée.
    """
    # On boucle jusqu'à l'avant-dernier segment pour pouvoir regarder le suivant
    for i in range(len(segments) - 1):
        segment_actuel = segments[i]
        segment_suivant = segments[i+1]

        # S'assurer que les segments ne sont pas vides
        if not segment_actuel or not segment_suivant:
            continue

        # Récupérer le dernier mot du segment actuel et le premier du suivant
        dernier_mot_actuel = segment_actuel[-1]
        premier_mot_suivant = segment_suivant[0]

        # Calculer le silence entre les deux segments
        temps_fin_actuel = dernier_mot_actuel['end']
        temps_debut_suivant = premier_mot_suivant['start']
        
        silence = temps_debut_suivant - temps_fin_actuel

        # Si le silence est positif mais inférieur au seuil, on ajuste
        if 0 < silence < seuil_secondes:
            print(f"Silence détecté ({silence:.2f}s) entre le segment {i} et {i+1} (inférieur à {seuil_secondes}s).")
            print(f"  - Modification de '{dernier_mot_actuel['word']}'. 'end' passe de {dernier_mot_actuel['end']} à {temps_debut_suivant}.")
            
            # On modifie la valeur 'end' pour qu'elle corresponde au début du segment suivant
            dernier_mot_actuel['end'] = temps_debut_suivant

    return segments

def format_srt_time(seconds):
    """Convertit un temps en secondes (float) au format SRT HH:MM:SS,ms."""
    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 1000)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    sec = total_seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{milliseconds:03d}"

def creer_srt_depuis_lignes(
    lignes_de_mots, 
    nom_fichier, 
    lignes_max_par_srt=2, 
    ponctuation_force_solo=True
):
    """
    Crée un fichier SRT en groupant des lignes de mots pré-définies.

    Args:
        lignes_de_mots (list): La liste des lignes. Chaque ligne est une liste de dicts.
        nom_fichier (str): Le nom du fichier de sortie.
        lignes_max_par_srt (int): Le nombre maximum de lignes à regrouper dans un sous-titre.
        ponctuation_force_solo (bool): Si True, une ligne terminant par .?! forme un sous-titre seul.
    """
    
    # --- Étape 1 : Regrouper les lignes en blocs de sous-titres ---
    blocs_de_lignes = []
    i = 0
    while i < len(lignes_de_mots):
        ligne_actuelle = lignes_de_mots[i]
        if not ligne_actuelle:
            i += 1
            continue
            
        texte_ligne = " ".join(d['word'] for d in ligne_actuelle)
        
        # Appliquer la règle de la ponctuation qui force un bloc solo
        if ponctuation_force_solo and texte_ligne.strip().endswith(('.', '?', '!')):
            blocs_de_lignes.append([ligne_actuelle])
            i += 1
        else:
            # Sinon, regrouper jusqu'à 'lignes_max_par_srt'
            fin_slice = i + lignes_max_par_srt
            groupe = lignes_de_mots[i:fin_slice]
            blocs_de_lignes.append(groupe)
            i += len(groupe)
    
    # --- Étape 2 : Formater les blocs en SRT ---
    srt_content = []
    for index, bloc in enumerate(blocs_de_lignes):
        if not bloc or not bloc[0]:
            continue

        # Le temps de début est celui du premier mot du bloc
        start_time = bloc[0][0]['start']
        # Le temps de fin est celui du dernier mot du bloc
        end_time = bloc[-1][-1]['end']
        
        # Construire le texte en joignant les lignes
        lignes_texte = [" ".join(d['word'] for d in ligne) for ligne in bloc]
        texte_formate = "\n".join(lignes_texte)

        # Assembler le bloc SRT complet
        srt_entry = (
            f"{index + 1}\n"
            f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n"
            f"{texte_formate}\n"
        )
        srt_content.append(srt_entry)

    # --- Étape 3 : Écrire le fichier ---
    with open(nom_fichier, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_content))
        
def create_app_structure_transcription_analysis():
    """
    Crée la structure de dossiers de l'application.
    """
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    path = base_path / "transcription_analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path

def main_smart_srt(audio_path, token, model="gpt-4.1-nano", user_prompt= "", max_caracteres = 60, seuil = 0.15, lignes_max_par_srt = 2, ponctuation_force_solo=True ):
    # 0. Transcription de l'audio
    config.API_STATUS = "Transcribing"
    transcription = groc_transcription(audio_path, token, user_prompt)
    transcription_segments = Get_Segments(transcription)
    transcription_segments = Set_Majuscule(transcription_segments)


    # 1. faire des batch des 10 segments
    transcription_segments_batch = []
    batch = []
    for segment in transcription_segments:
        batch.append(segment)
        if len(batch) == 10:
            transcription_segments_batch.append(batch)
            batch = []
    if batch:
        transcription_segments_batch.append(batch)


    # 2. Corriger chacun des batchs
    for batch, i in zip(transcription_segments_batch, range(len(transcription_segments_batch))):
        # batch = transcription_segments_batch[0]
        config.API_STATUS = f"Spelling Correction {i}/{len(transcription_segments_batch)} "
        segment_text = [_afficher_phrase(segment, preview=False) for segment in batch]
        segments_text_corrected = Spelling_Correction(segment_text, user_prompt, model, token)
        batch = synchroniser_corrections(batch, segments_text_corrected)

        
    #3. Regrouper les batchs corrigés
    transcription_segments_corrected = []
    for batch in transcription_segments_batch:
        transcription_segments_corrected.extend(batch)


    #4. Si un batch est juste un mot avec une virgule, le rattacher au suivant
    config.API_STATUS = "Checking Punctuation"
    for i in range(len(transcription_segments_corrected) - 2):
        
        current_segment = transcription_segments_corrected[i]
        try :
            next_segment = transcription_segments_corrected[i + 1]
        except IndexError:
            break
        if len(current_segment) == 1 and current_segment[0]['word'][-1] == ',':
            transcription_segments_corrected[i+1].insert(0, current_segment[0])
            transcription_segments_corrected.pop(i)


    #5. Séparation en bloc de bonne taille
    transcription_segments_1 = []
    for segment, i in zip(transcription_segments_corrected, range(len(transcription_segments_corrected))):
        config.API_STATUS = f"Generating Smart Subtitles {i}/{len(transcription_segments_corrected)} "
        segments_splited = decouper_segment_intelligent(segment, max_caracteres)
        transcription_segments_1.extend(segments_splited)

        # for i, bloc in enumerate(segments_splited):
        #     texte_bloc = " ".join([d['word'] for d in bloc])
        #     start_time = bloc[0]['start']
        #     end_time = bloc[-1]['end']
        #     print(f"  Bloc {i+1} (de {start_time}s à {end_time}s) : '{texte_bloc}'")

        # print("\n" + "="*50 + "\n")
        

    # 6. Ajuster les silences entre les segments

    config.API_STATUS = "Adjusting Silences"
    transcription_segments_2 = ajuster_silences_inter_segments(transcription_segments_1, seuil)


    # 7. Export des SRT

    config.API_STATUS = "Exporting SRT"
    chaine_de_temps = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_path = os.path.join(create_app_structure_transcription_analysis(), f"SRT_{chaine_de_temps}.srt")
    creer_srt_depuis_lignes(
        transcription_segments_2,
        output_path,
        lignes_max_par_srt,
        ponctuation_force_solo
    )

    return output_path





def find_passages(segments_list, prompt , model, token):

    # model = "gemini-2.5-flash-preview-04-17"  
    # model = "gemini-2.5-pro-preview-05-06"  

    prompt = f"""

    ### Instructions:
    - You are an expert assistant for finding passages in a transcription.
    - You are given a transcription and a user request.
    - You need to return the passages in the transcription that are relevant to the user request.

    ### User request to find passages :
    {prompt}
    
    ### Transcription to use :
    {segments_list}
        """.strip()
        
    try:
        response = requests.post(
            f"{config.API_URL}/global-gemini-call",
            json={"prompt": prompt,
                  "model": model,
                  "schema_type" : "audio_search"
                  },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Erreur lors de l'amélioration du prompt: {response.status_code}")
            return []
            
        return response.json()["passages"]
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API d'amélioration de prompt: {str(e)}")
        return []

def creer_srt_depuis_lignes_passages(
    lignes_de_mots, 
    lignes_max_par_srt=2, 
    ponctuation_force_solo=True
):
    """
    Crée un fichier SRT en groupant des lignes de mots pré-définies.

    Args:
        lignes_de_mots (list): La liste des lignes. Chaque ligne est une liste de dicts.
        nom_fichier (str): Le nom du fichier de sortie.
        lignes_max_par_srt (int): Le nombre maximum de lignes à regrouper dans un sous-titre.
        ponctuation_force_solo (bool): Si True, une ligne terminant par .?! forme un sous-titre seul.
    """
    
    # --- Étape 1 : Regrouper les lignes en blocs de sous-titres ---
    blocs_de_lignes = []
    i = 0
    while i < len(lignes_de_mots):
        ligne_actuelle = lignes_de_mots[i]
        if not ligne_actuelle:
            i += 1
            continue
            
        texte_ligne = " ".join(d['word'] for d in ligne_actuelle)
        
        # Appliquer la règle de la ponctuation qui force un bloc solo
        if ponctuation_force_solo and texte_ligne.strip().endswith(('.', '?', '!')):
            blocs_de_lignes.append([ligne_actuelle])
            i += 1
        else:
            # Sinon, regrouper jusqu'à 'lignes_max_par_srt'
            fin_slice = i + lignes_max_par_srt
            groupe = lignes_de_mots[i:fin_slice]
            blocs_de_lignes.append(groupe)
            i += len(groupe)
    
    # --- Étape 2 : Formater les blocs en SRT ---
    srt_content = []
    for index, bloc in enumerate(blocs_de_lignes):
        if not bloc or not bloc[0]:
            continue

        # Le temps de début est celui du premier mot du bloc
        start_time = bloc[0][0]['start']
        # Le temps de fin est celui du dernier mot du bloc
        end_time = bloc[-1][-1]['end']
        
        # Construire le texte en joignant les lignes
        lignes_texte = [" ".join(d['word'] for d in ligne) for ligne in bloc]
        texte_formate = "\n".join(lignes_texte)

        # Assembler le bloc SRT complet
        srt_entry = (
            f"{index + 1}\n"
            f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n"
            f"{texte_formate}\n"
        )
        srt_content.append(srt_entry)

    # --- Étape 3 : Écrire le fichier ---
    # with open(nom_fichier, 'w', encoding='utf-8') as f:
    #     f.write("\n".join(srt_content))
    return srt_content
 
def groc_transcription_passages(audio_path, token):

    def encode_audio_base64(audio_path):
        with open(audio_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded

    audio_base64 = encode_audio_base64(audio_path)
    
    # ====== Remplacé dans l'API

    filename = os.path.basename(audio_path)
    url = f"{config.API_URL}/speaker_analysis"  
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"  
    }
    
    payload = {
        "audio_base64": audio_base64,
        "filename": filename
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  
    
    result = response.json()
    transcription = result["transcription"]


    # ======


    return transcription
 
def main_find_passages(audio_path, token, model="gemini-2.5-pro-preview-05-06", user_prompt= "", max_caracteres = 120, seuil = 0.15, lignes_max_par_srt = 2, ponctuation_force_solo=True ):
    # 0. Transcription de l'audio
    config.API_STATUS = "Transcribing"
    transcription = groc_transcription_passages(audio_path, token)
    transcription_segments = Get_Segments(transcription)
    transcription_segments = Set_Majuscule(transcription_segments)


    # 1. faire des batch des 10 segments
    transcription_segments_batch = []
    batch = []
    for segment in transcription_segments:
        batch.append(segment)
        if len(batch) == 10:
            transcription_segments_batch.append(batch)
            batch = []
    if batch:
        transcription_segments_batch.append(batch)

        
    #3. Regrouper les batchs corrigés
    transcription_segments_corrected = []
    for batch in transcription_segments_batch:
        transcription_segments_corrected.extend(batch)


    #4. Si un batch est juste un mot avec une virgule, le rattacher au suivant
    config.API_STATUS = "Checking Punctuation"
    for i in range(len(transcription_segments_corrected) - 2):
        
        current_segment = transcription_segments_corrected[i]
        try :
            next_segment = transcription_segments_corrected[i + 1]
        except IndexError:
            break
        if len(current_segment) == 1 and current_segment[0]['word'][-1] == ',':
            transcription_segments_corrected[i+1].insert(0, current_segment[0])
            transcription_segments_corrected.pop(i)


    #5. Séparation en bloc de bonne taille
    transcription_segments_1 = []
    for segment, i in zip(transcription_segments_corrected, range(len(transcription_segments_corrected))):
        segments_splited = decouper_segment_intelligent(segment, max_caracteres)
        transcription_segments_1.extend(segments_splited)


    # 7. Export des SRT
    segments_srt = creer_srt_depuis_lignes_passages(
        transcription_segments_1,
        lignes_max_par_srt,
        ponctuation_force_solo
    )

    return find_passages(segments_srt, user_prompt, model, token)




