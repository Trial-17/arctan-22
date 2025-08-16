import cv2
import numpy as np
import base64
import requests
import time
from LIB import config
from datetime import datetime
from pathlib import Path

def create_app_structure_image_generation():
    """
    Crée la structure de dossiers de l'application.
    """
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    image_generation_path = base_path / "image_generation"
    image_generation_path.mkdir(parents=True, exist_ok=True)
    return image_generation_path

# def encode_image_to_base64_with_alpha_mask(image_path):
#     """
#     Charge une image, transforme le noir absolu en alpha 0 (transparent),
#     et encode le résultat en base64 PNG avec canal alpha.
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     # Forcer en 3 canaux si image en BGR uniquement
#     if image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

#     # Créer le canal alpha : 0 si noir absolu, 255 sinon
#     black_pixels = np.all(image[:, :, :3] == [0, 0, 0], axis=2)
#     alpha_channel = np.where(black_pixels, 0, 255).astype(np.uint8)
#     image[:, :, 3] = alpha_channel

#     # Encodage en PNG avec alpha
#     success, buffer = cv2.imencode('.png', image)
#     if not success:
#         raise ValueError("Failed to encode image to PNG")

#     base64_str = base64.b64encode(buffer).decode('utf-8')
#     return base64_str

def poll_flux_result(request_id, max_wait=180):
    """
    Poll l'API Flux Fill Pro jusqu'à ce que le résultat soit prêt.
    
    :param request_id: ID retourné par la requête initiale
    :param max_wait: Durée maximale d'attente en secondes
    :return: URL signée pour récupérer l'image générée
    """
    start_time = time.time()

    while True:
        if time.time() - start_time > max_wait:
            raise TimeoutError("Polling timed out")

        time.sleep(0.5)
        response = requests.get(
            'https://api.us1.bfl.ai/v1/get_result',
            headers={
                'accept': 'application/json',
            },
            params={'id': request_id}
        )
        response.raise_for_status()
        result = response.json()

        status = result.get("status")
        if status == "Ready":
            return result["result"]["sample"]  
        else:
            print(f"Status: {status}")

def download_and_save_image_from_url(url, output_folder):
    response = requests.get(url)
    response.raise_for_status()
    filename = datetime.now().strftime("fill_%y%m%d%H%M%S.png")
    output_path = output_folder / filename
    with open(output_path, "wb") as f:
        f.write(response.content)
    return str(output_path)

def call_flux_fill(token, image_b64, prompt=""):

    url = f"{config.API_URL}/flux-fill"
    headers = {

        'Authorization': f'Bearer {token}',
        "Content-Type": "application/json"
    }

    payload = {
        "base64_image": image_b64,
        "prompt": prompt
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["result"]

def call_flux_extend(token, borders, image_b64, prompt=""):

    url = f"{config.API_URL}/flux-extend"
    headers = {

        'Authorization': f'Bearer {token}',
        "Content-Type": "application/json"
    }

    payload = {
        "base64_image": image_b64,
        "prompt": prompt,
        "borders": borders
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["result"]

# def detect_black_borders_and_crop_base64(image_path):
#     """
#     Détecte les bordures noires absolues, retourne leur taille et l'image recadrée en base64 PNG.
#     :param image_path: Chemin de l’image
#     :return: dict avec top/bottom/left/right et l'image croppée encodée en base64
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     # Masque des pixels noir absolu
#     black_mask = np.all(image == [0, 0, 0], axis=2).astype(np.uint8)
#     height, width = black_mask.shape

#     # Détection bordures noires
#     top = bottom = left = right = 0

#     for row in range(height):
#         if np.all(black_mask[row, :] == 1):
#             top += 1
#         else:
#             break

#     for row in range(height - 1, -1, -1):
#         if np.all(black_mask[row, :] == 1):
#             bottom += 1
#         else:
#             break

#     for col in range(width):
#         if np.all(black_mask[:, col] == 1):
#             left += 1
#         else:
#             break

#     for col in range(width - 1, -1, -1):
#         if np.all(black_mask[:, col] == 1):
#             right += 1
#         else:
#             break

#     # Recadrage de l'image
#     cropped_image = image[top:height - bottom, left:width - right]

#     # Encodage en base64
#     _, buffer = cv2.imencode('.png', cropped_image)
#     base64_image = base64.b64encode(buffer).decode('utf-8')

#     return {
#         "borders": {"top": top, "bottom": bottom, "left": left, "right": right},
#         "image_base64": base64_image
#     }

def detect_transparent_borders_and_crop_base64(image_path):
    """
    Détecte les bordures transparentes (canal alpha), retourne leur taille 
    et l'image recadrée encodée en base64 PNG.
    
    :param image_path: Chemin de l’image (doit être un format supportant la transparence, comme PNG).
    :return: dict avec top/bottom/left/right et l'image croppée encodée en base64.
    """
    # ÉTAPE 1: Lire l'image en conservant le canal alpha (IMREAD_UNCHANGED)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise FileNotFoundError(f"Image non trouvée ou format non supporté : {image_path}")

    # Vérifier si l'image a bien un canal alpha
    if image.shape[2] < 4:
        print("L'image ne possède pas de canal alpha. Retour de l'image originale.")
        # Pas de canal alpha, donc pas de bordures transparentes. On retourne l'image telle quelle.
        _, buffer = cv2.imencode('.png', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return {
            "borders": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "image_base64": base64_image
        }

    # ÉTAPE 2: Créer un masque basé sur le canal alpha
    # Le canal alpha est le 4ème canal (index 3)
    alpha_channel = image[:, :, 3]
    # Le masque est vrai (1) là où l'alpha est 0 (totalement transparent)
    transparent_mask = (alpha_channel == 0).astype(np.uint8)
    
    height, width = transparent_mask.shape

    # Le reste de la logique de détection est identique, mais utilise le nouveau masque
    top = bottom = left = right = 0

    for row in range(height):
        if np.all(transparent_mask[row, :] == 1):
            top += 1
        else:
            break

    for row in range(height - 1, -1, -1):
        if np.all(transparent_mask[row, :] == 1):
            bottom += 1
        else:
            break

    for col in range(width):
        if np.all(transparent_mask[:, col] == 1):
            left += 1
        else:
            break

    for col in range(width - 1, -1, -1):
        if np.all(transparent_mask[:, col] == 1):
            right += 1
        else:
            break

    # Recadrage de l'image (en conservant tous les canaux, y compris l'alpha)
    cropped_image = image[top:height - bottom, left:width - right]

    # Encodage en base64 (le format PNG préserve la transparence)
    _, buffer = cv2.imencode('.png', cropped_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return {
        "borders": {"top": top, "bottom": bottom, "left": left, "right": right},
        "image_base64": base64_image
    }


def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_image_with_flux(token,image_path, prompt=""):

    image_generation_folder = create_app_structure_image_generation()
    
    border_result = detect_transparent_borders_and_crop_base64(image_path)

    if border_result["borders"]["top"] <= 5 and border_result["borders"]["bottom"] <= 5 and border_result["borders"]["left"] <= 5 and border_result["borders"]["right"] <= 5:
        print("No black borders detected.")
        image_b64 = encode_image_base64(image_path)
        data = call_flux_fill(token, image_b64, prompt)
    else: 
        image_b64 = border_result["image_base64"]
        borders = border_result["borders"]
        print(f"Borders detected: {borders}")
        data = call_flux_extend(token, borders, image_b64, prompt)
        
    request_id = data.get("id") 
    encoded_result = poll_flux_result(request_id)
    path = download_and_save_image_from_url(encoded_result, image_generation_folder)
    return path
