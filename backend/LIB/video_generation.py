import os
from pathlib import Path
import time
import base64
from LIB import config
import requests
from PIL import Image




# ------- Fonctions Video Generation

def optimize_prompt(prompt, token):

    
    response = requests.post(
        f"{config.API_URL}/optimize-video-prompt",
        json={"prompt": prompt},
        headers={"Authorization": f"Bearer {token}"}
    )
    response.raise_for_status()  # Lève une exception si la requête échoue
    prompt_optimized = response.json()["prompt_optimized"]
    return prompt_optimized

def create_app_structure_image_generation():
    """
    Crée la structure de dossiers de l'application.
    """
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    image_generation_path = base_path / "image_generation"
    image_generation_path.mkdir(parents=True, exist_ok=True)
    return image_generation_path

def create_app_structure_image_generation():
    """
    Crée la structure de dossiers de l'application.
    """
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    image_generation_path = base_path / "image_generation"
    image_generation_path.mkdir(parents=True, exist_ok=True)
    return image_generation_path

def compress_image_if_needed(image_path):
    # Vérifications de base
    if not image_path or not isinstance(image_path, str) or not os.path.exists(image_path):
        print("❌ Chemin de l'image invalide ou fichier non trouvé.")
        return

    max_size_mb = 3
    max_size_bytes = max_size_mb * 1024 * 1024
    min_ratio, max_ratio = 0.5, 2.0

    # Ouvre et convertit en RGB si nécessaire
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        ratio = width / height

        # Recadrage si ratio hors limites
        if ratio < min_ratio:
            # Trop étroit (trop haut) : on réduit la hauteur
            target_height = int(width / min_ratio)
            top = (height - target_height) // 2
            img = img.crop((0, top, width, top + target_height))

        elif ratio > max_ratio:
            # Trop large : on réduit la largeur
            target_width = int(height * max_ratio)
            left = (width - target_width) // 2
            img = img.crop((left, 0, left + target_width, height))

        # Sauvegarde du recadrage (même si pas modifié, ça réécrit l’image au format JPEG optimisé)
        img.save(image_path, format='JPEG', quality=95, optimize=True)

    # Vérifie la taille après recadrage
    size_bytes = os.path.getsize(image_path)
    if size_bytes <= max_size_bytes:
        return  # Pas besoin de compresser

    # Ouvre à nouveau pour la compression progressive
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        tmp_path = image_path + ".tmp.jpg"
        for quality in range(85, 30, -5):
            img.save(tmp_path, format='JPEG', quality=quality, optimize=True)
            if os.path.getsize(tmp_path) <= max_size_bytes:
                break
        else:
            os.remove(tmp_path)
            raise ValueError("Impossible de compresser l'image sous 3 Mo.")

        # Remplace l'original par la version compressée
        os.replace(tmp_path, image_path)

def submit_runway_request(token, base64_image, prompt_text, ratio="1584:672"):
    config.API_STATUS = "Submitting"
    
    # Utilisation de l'endpoint relay au lieu de l'appel direct à l'API
    response = requests.post(
        f"{config.API_URL}/runway-video",
        json={
            "base64_image": base64_image,
            "prompt_text": prompt_text,
            "ratio": ratio
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code != 200:
        raise Exception(f"Erreur lors de la génération de vidéo: {response.text}")
    
    config.API_STATUS = "Running"
    
    # Récupérer l'URL de la vidéo générée
    result = response.json()
    video_url = result["video_url"]
    
    return video_url

def download_video(video_url, output_dir, filename=None):
    # Générer un nom de fichier unique basé sur le timestamp
    config.API_STATUS = "Downloading"
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"video_generation_{timestamp}.mp4"
    
    output_path = output_dir / filename
    response = requests.get(video_url, stream=True)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ Vidéo téléchargée : {output_path}")
        return str(output_path)
    else:
        raise Exception(f"Échec du téléchargement. Code {response.status_code}")
    
def generate_runway_video(token, image_path, prompt, ratio="1584:672", type_rush  = "Rush"):
    
    # compress image if needed
    compress_image_if_needed(image_path)
    
    # optimize prompt
    prompt_text = optimize_prompt(prompt,  token)
    
    # encode image to base64
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    # Mapping Type Rush 
    if type_rush == "Rush" : prompt_text += ""
    else : prompt_text += "Goal : effect creation. Focus on the effect, keep the scene realistic, simply add what you need for matching the desire effect"
   
    # respecter la taille max 
    prompt_text = prompt_text[:999]
    
    video_url = submit_runway_request(token, base64_image, prompt_text, ratio)
    
    output_dir = create_app_structure_image_generation()
    output_path =  download_video(video_url, output_dir)
    
    return output_path


