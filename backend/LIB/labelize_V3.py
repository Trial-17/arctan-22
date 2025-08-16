# import json
# from pathlib import Path
# import base64
# import concurrent.futures
# import os
# import base64

# import concurrent.futures
# import av
# from PIL import Image
# import cv2
# import numpy as np
# from pymediainfo import MediaInfo

# from LIB import config


# def create_app_structure():
#     """
#     Crée la structure de dossiers de l'application dans Documents/Adobe/Premiere Pro/Premiere Copilot/
#     Retourne un dictionnaire contenant les chemins de tous les dossiers créés.
#     """
#     # Chemin de base
#     base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    
#     # Structure des dossiers
#     folders = {
#         "thumbnails": str(base_path / "thumbnails"),
#         "temp": str(base_path / "temp"),
#         "rush_db": str(base_path / "rush_db"),

#     }
    
#     # Création des dossiers
#     for folder_path in folders.values():
#         Path(folder_path).mkdir(parents=True, exist_ok=True)
    
#     return folders



# GLOBAL_DB_PATH = str( create_app_structure()['rush_db'] + "/rush_db.json")
# GLOBAL_TEMP_FOLDER = str(create_app_structure()['temp'])
# GLOBAL_THUMBNAILS_FOLDER = str(create_app_structure()['thumbnails'])



# # ------- Fonctions

# def extract_frames_pyav(inputs):
#     """Extrait 2 images espacées d'un fichier ou d'un dossier et retourne la liste des fichiers analysés en enregistrant les frames en 1080p."""
    
#     # Vérifier que temp_folder existe
#     os.makedirs(GLOBAL_TEMP_FOLDER, exist_ok=True)
#     os.makedirs(GLOBAL_THUMBNAILS_FOLDER, exist_ok=True)

#     # Récupérer la liste des fichiers à traiter
#     video_files = []

#     # Traiter chaque élément de la liste d'entrée
#     for input_path in inputs:
#         if os.path.isfile(input_path):  # C'est un fichier unique
#             if input_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv')):
#                 video_files.append(input_path)
#         elif os.path.isdir(input_path):  # C'est un dossier
#             for root, _, files in os.walk(input_path):
#                 for file in files:
#                     if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv')):
#                         video_files.append(os.path.join(root, file))

#     analyzed_files = []  # Liste des fichiers traités

#     def process_video(video_path):
#         """Extrait 4 images espacées d'une vidéo avec PyAV et les redimensionne en 1080p."""
#         try:
#             container = av.open(video_path)
#             stream = container.streams.video[0]

#             total_frames = stream.frames
#             duration = stream.duration / stream.time_base  # Durée en secondes

#             timestamps = [duration * 0.25, duration * 0.75]
#             video_name = os.path.splitext(os.path.basename(video_path))[0]

#             extracted_images = []

#             # Pour la première frame uniquement, créer une thumbnail
#             first_frame = True
            
#             for i, timestamp in enumerate(timestamps):
#                 container.seek(int(timestamp * stream.time_base), stream=stream)

#                 # Lire plusieurs frames après le seek pour éviter les duplications
#                 for frame in container.decode(video=0):
#                     if frame:
#                         output_filename = os.path.join(GLOBAL_TEMP_FOLDER, f"{video_name}_frame{i+1}.jpg")

#                         # Convertir en image PIL et redimensionner en 1080p
#                         img = frame.to_image()
#                         img_resized = img.resize((1920, 1080), Image.LANCZOS)
#                         img_resized.save(output_filename, "JPEG", quality=95)
                        
#                         # Pour la première frame, créer une thumbnail avec le nom encodé
#                         if first_frame:
#                             # Encoder le chemin en base64
#                             encoded_path = base64.b64encode(video_path.encode()).decode()
#                             thumbnail_filename = os.path.join(GLOBAL_THUMBNAILS_FOLDER, f"{encoded_path}.jpg")
#                             img_resized.save(thumbnail_filename, "JPEG", quality=70)
#                             first_frame = False

#                         extracted_images.append(img_resized)
#                         break  # Prendre uniquement la première frame différente

#             analyzed_files.append(video_path)  # Ajouter à la liste des fichiers traités

#         except Exception:
#             pass  # Ignore les erreurs et continue

#     # Lancer l'extraction pour toutes les vidéos trouvées
#     cpt = 1
#     for video in video_files:
#         process_video(video)
#         config.API_STATUS = "Extraction : " + str(cpt) + "/" + str(len(video_files))
#         cpt += 1
#     return analyzed_files  # Retourner la liste des fichiers analysés

# def load_existing_rush_paths():
#     """Charge les rushs déjà présents dans le fichier global et retourne leurs chemins."""
#     if os.path.exists(GLOBAL_DB_PATH):
#         try:
#             with open(GLOBAL_DB_PATH, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 return {item["file"] for item in data}  # Ensemble des rushs déjà enregistrés
#         except json.JSONDecodeError:
#             return set()  # Retourne un ensemble vide si le JSON est corrompu
#     return set()  # Retourne un ensemble vide si le fichier n'existe pas

# def filter_new_rushes(rush_paths):
#     """Supprime de rush_paths les rushs déjà présents dans le fichier global."""
#     existing_rushes = load_existing_rush_paths()
#     new_rushes = [rush for rush in rush_paths if rush not in existing_rushes]
#     final_rushes = []   
#     for input_path in new_rushes:
#         if os.path.isfile(input_path):  # C'est un fichier unique
#             if input_path.lower().endswith(('.mp4', '.mov', '.avi')):
#                 final_rushes.append(input_path)
#         elif os.path.isdir(input_path):  # C'est un dossier
#             for root, _, files in os.walk(input_path):
#                 for file in files:
#                     if file.lower().endswith(('.mp4', '.mov', '.avi')):
#                         final_rushes.append(os.path.join(root, file))
    
    
#     return final_rushes

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

# def labelize_image(image_path, api_key, mode = 'low'):
    
#     base64_image = encode_image(image_path)
    
#     try:
#         import requests
#         import json
        
#         # URL de l'API
#         api_url = f"{config.API_URL}/labelize-image"  # Ajustez l'URL selon votre configuration
        
#         # Headers pour l'authentification
#         headers = {
#             "Authorization": f"Bearer {api_key}"
#         }
        
#         # Données de la requête
#         data = {
#             "base64_image": base64_image,
#             "mode": mode
#         }
        
#         # Appel à l'API
#         response = requests.post(api_url, json=data, headers=headers)
        
#         # Vérifier si la requête a réussi
#         response.raise_for_status()
        
#         # Retourner les données

#         return response.json()
        
#     except Exception as e:
#         print(f"Erreur lors de l'appel à l'API de labellisation: {str(e)}")
#         raise

# def get_images_from_folder(folder):
#     """Récupère toutes les images .jpg se terminant par '_frame1' dans un dossier et ses sous-dossiers"""
#     images = []
#     for root, _, files in os.walk(folder):
#         for file in files:
#             if file.lower().endswith("_frame1.jpg"):
#                 images.append(os.path.join(root, file))
#     return images

# def process_image(image_path, api_key):
#     """Labelise une image et retourne la réponse de l'API"""
#     try:
#         return labelize_image(image_path, api_key)
#     except Exception:
#         return None

# def find_matching_rush(image_path, rush_paths):
#     """Trouve le fichier rush correspondant à une image `_frame1.jpg`"""
#     base_name = os.path.basename(image_path).rsplit("_frame1.jpg", 1)[0]  # Ex: "DJI_0751"
#     for rush in rush_paths:
#         if os.path.basename(rush).startswith(base_name):
#             return rush  # Retourne le rush correspondant
#     return None  # Si aucun rush ne correspond

# def process_images_in_folder(input_folder, rush_paths, api_key, max_workers=5):
#     """Traite toutes les images d'un dossier en parallèle et retourne les résultats liés aux rushs"""
#     images = get_images_from_folder(input_folder)
    
#     results = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {executor.submit(process_image, img, api_key): img for img in images}
#         for future in concurrent.futures.as_completed(futures):
#             result = future.result()
#             img_path = futures[future]  # Récupérer le chemin de l'image traitée
#             if result:
#                 rush_path = find_matching_rush(img_path, rush_paths)  # Trouver le rush correspondant
#                 if rush_path:
#                     results.append({"file": rush_path, "data": result})

#     return results

# def get_image_pairs(image_folder):
#     """Récupère les paires de frames (frame1.jpg, frame2.jpg) dans un dossier."""
#     image_pairs = []
#     all_images = sorted(os.listdir(image_folder))  # Trier les fichiers pour éviter des erreurs d'ordre
    
#     for img in all_images:
#         if img.endswith("_frame1.jpg"):
#             base_name = img.replace("_frame1.jpg", "")
#             frame2 = base_name + "_frame2.jpg"
#             if frame2 in all_images:
#                 image_pairs.append((os.path.join(image_folder, img), os.path.join(image_folder, frame2)))

#     return image_pairs

# def detect_camera_motion(image1_path, image2_path, threshold_translation=5, threshold_zoom=0.02, display=False):
#     """Détecte les mouvements de caméra entre deux frames."""
    
#     # Charger les images en niveaux de gris
#     image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
#     if image1 is None or image2 is None:
#         return None  # Évite de planter si une image est manquante

#     # Détecter les points clés avec ORB
#     orb = cv2.ORB_create(nfeatures=1000)
#     kp1, des1 = orb.detectAndCompute(image1, None)
#     kp2, des2 = orb.detectAndCompute(image2, None)

#     if des1 is None or des2 is None:
#         return None  # Pas assez de points détectés

#     # Matcher les points clés entre les deux images
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
#     matches = bf.match(des1, des2)
#     matches = sorted(matches, key=lambda x: x.distance)[:50]  # Prendre les 50 meilleures correspondances

#     if len(matches) < 4:
#         return None  # Pas assez de points pour une estimation fiable

#     # Extraire les points correspondants
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

#     # Trouver l'homographie
#     H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     if H is None:
#         return None  # Échec de l'estimation de l'homographie

#     # Extraire la translation et le zoom
#     dx, dy = H[0, 2], H[1, 2]  # Translation (pan/tilt)
#     scale_x = np.linalg.norm(H[0, :2])  # Échelle sur X
#     scale_y = np.linalg.norm(H[1, :2])  # Échelle sur Y
#     zoom = (scale_x + scale_y) / 2  # Moyenne des échelles

#     detected_movements = []

#     # Détection de PAN (mouvement horizontal)
#     if abs(dx) > threshold_translation:
#         direction_x = "right" if dx > 0 else "left"
#         detected_movements.append(f"Pan {direction_x}")

#     # Détection de TILT (mouvement vertical)
#     if abs(dy) > threshold_translation:
#         direction_y = "up" if dy > 0 else "down"
#         detected_movements.append(f"Tilt {direction_y}")

#     # Détection de ZOOM
#     if abs(zoom - 1.0) > threshold_zoom:
#         zoom_type = "zoom in" if zoom > 1.0 else "zoom out"
#         detected_movements.append(zoom_type)

#     # Affichage si demandé
#     if display:

#         print(f"Mouvements détectés : {detected_movements}", dx, dy)

#     return detected_movements if detected_movements else ["no motion"]

# def analyze_camera_motion(image_folder, results, threshold_translation=5, threshold_zoom=0.02, display=False):
#     """Analyse les mouvements de caméra et met à jour les résultats de labelisation."""
#     image_pairs = get_image_pairs(image_folder)
#     cpt = 1
#     for frame1, frame2 in image_pairs:
#         base_name = os.path.basename(frame1).replace("_frame1.jpg", "")  # Ex: "DJI_0751"
#         movements = detect_camera_motion(frame1, frame2, threshold_translation, threshold_zoom, display)

#         # Associer à son fichier rush
#         for item in results:
#             rush_filename = os.path.basename(item["file"])  # Ex: "DJI_0751.MP4"
#             rush_base_name = os.path.splitext(rush_filename)[0]  # Enlever ".MP4"

#             if rush_base_name == base_name:  # Vérifier si le rush correspond aux frames
#                 item["data"]["camera_motion"] = movements
#                 break  # Sortir dès qu'on a mis à jour l'élément correspondant
#         config.API_STATUS = "Motion : " + str(cpt) + "/" + str(len(image_pairs))
#         cpt += 1
#     return results

# def get_video_duration(video_path):
#     """Retourne la durée d'une vidéo en secondes en utilisant OpenCV."""
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return None  # Impossible d'ouvrir la vidéo
        
#         fps = cap.get(cv2.CAP_PROP_FPS)  # Récupérer le nombre de frames par seconde
#         total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Nombre total de frames
#         cap.release()
        
#         if fps > 0 and total_frames > 0:
#             return total_frames / fps  # Durée en secondes
#         return None
#     except Exception:
#         return None  # En cas d'erreur

# def add_video_durations_to_labels(labels):
#     """Ajoute la durée de chaque rush aux labels en se basant sur 'file'."""
#     cpt = 1
#     for label in labels:
#         video_path = label.get("file")  # Récupère le chemin du rush
#         if video_path:  # Vérifie que le chemin est valide
#             duration = get_video_duration(video_path)
#             if duration is not None:
#                 label["data"]["duration"] = duration  # Ajoute la durée
#         config.API_STATUS = "Duration : " + str(cpt) + "/" + str(len(labels))
#         cpt += 1

#     return labels

# def load_global_labels():
#     """Charge les labels existants depuis le fichier global."""
#     if os.path.exists(GLOBAL_DB_PATH):
#         try:
#             with open(GLOBAL_DB_PATH, "r", encoding="utf-8") as f:
#                 return json.load(f)  # Charger les données existantes
#         except json.JSONDecodeError:
#             return []  # Retourne une liste vide si le JSON est corrompu
#     return []  # Si le fichier n'existe pas, retourne une liste vide

# def save_global_labels(labels):
#     """Sauvegarde les labels mis à jour dans le fichier global."""
#     with open(GLOBAL_DB_PATH, "w", encoding="utf-8") as f:
#         json.dump(labels, f, indent=4, ensure_ascii=False)  # Écriture propre du JSON

# def update_global_labels(new_labels):
#     """Ajoute de nouveaux labels au fichier global sans doublons."""
#     existing_labels = load_global_labels()
    
#     # Convertir la liste existante en dictionnaire basé sur le 'file' pour éviter les doublons
#     labels_dict = {label["file"]: label for label in existing_labels}

#     # Ajouter / mettre à jour les nouveaux labels
#     for label in new_labels:
#         labels_dict[label["file"]] = label  # Remplace si le fichier existe déjà

#     # Sauvegarde des labels mis à jour
#     save_global_labels(list(labels_dict.values()))

# def get_thumbnail_path(labels):
#     cpt = 1
#     for label in labels: 
#         video_path = label.get("file")
#         if video_path:  # Vérifie que le chemin est valide
#             encoded_path = base64.b64encode(video_path.encode()).decode()
#             thumbnail_path = os.path.join(GLOBAL_THUMBNAILS_FOLDER, f"{encoded_path}.jpg")
#             if os.path.exists(thumbnail_path):
#                 label["thumbnail"] = thumbnail_path
#             else :
#                 label["thumbnail"] = None
#         config.API_STATUS = "Thumbnail : " + str(cpt) + "/" + str(len(labels))
#         cpt+=1
#     return labels

# def get_video_metadata(labels):
#     cpt = 1
#     for label in labels:
#         video_path = label.get("file")
#         if video_path:
#             media_info = MediaInfo.parse(video_path)
#             for track in media_info.tracks:
#                 if track.track_type == "General":
#                     label["Date"] = track.tagged_date
#                     label["Format"] = track.format
#                 if track.track_type == "Video":
#                     label["Resolution"] = f"{track.width}x{track.height}"
#                     label["Frame rate"] = track.frame_rate
        
#         config.API_STATUS = "Metadata : " + str(cpt) + "/" + str(len(labels))
#         cpt +=1
#     return labels

# def main_labelize(file_paths, token):
    
#     GLOBAL_DB_PATH = str( create_app_structure()['rush_db'] + "/rush_db.json")
#     GLOBAL_TEMP_FOLDER = str(create_app_structure()['temp'])
#     GLOBAL_THUMBNAILS_FOLDER = str(create_app_structure()['thumbnails'])

#     rush_paths = filter_new_rushes(file_paths)
#     rush_paths = extract_frames_pyav(rush_paths)
#     if len(rush_paths) == 0:
#         print("No new rushes found.")
#         return []
    
#     config.API_STATUS = "Starting Labelization"

#     label = process_images_in_folder(GLOBAL_TEMP_FOLDER, rush_paths, token, 2)
    
#     label = analyze_camera_motion(GLOBAL_TEMP_FOLDER, label, threshold_translation=600, threshold_zoom=5, display=False)
#     label = add_video_durations_to_labels(label)
#     label = get_thumbnail_path(label)
#     label = get_video_metadata(label)
#     update_global_labels(label)


#     # for file in os.listdir(GLOBAL_TEMP_FOLDER):
#     #     file_path = os.path.join(GLOBAL_TEMP_FOLDER, file)
#     #     try:
#     #         if os.path.isfile(file_path):
#     #             os.unlink(file_path)
#     #     except Exception as e:
#     #         pass


    
#     return rush_paths
 