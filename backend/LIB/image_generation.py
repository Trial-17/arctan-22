import os

import base64
import requests
import time
from datetime import datetime
from pathlib import Path
from LIB import config





def get_enhanced_prompt(prompt, token):
    """
    Appelle l'API externe pour améliorer le prompt de génération d'image.
    """
    try:
        response = requests.post(
            f"{config.API_URL}/enhance-prompt",
            json={"prompt": prompt},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Erreur lors de l'amélioration du prompt: {response.status_code}")
            return {"enhancedPrompt": prompt}  # En cas d'erreur, retourne le prompt original
            
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API d'amélioration de prompt: {str(e)}")
        return {"enhancedPrompt": prompt}  # En cas d'erreur, retourne le prompt original

def create_app_structure_image_generation():
    """
    Crée la structure de dossiers de l'application.
    """
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    image_generation_path = base_path / "image_generation"
    image_generation_path.mkdir(parents=True, exist_ok=True)
    return image_generation_path

def call_flux_pro_ultra(token, base64_image, aspect = "21:9", prompt="an image"):
    
    url = f"{config.API_URL}/generate-image-from-flux"
    headers = {

        'Authorization': f'Bearer {token}',
        "Content-Type": "application/json"
    }

    payload = {
        "base64_image": base64_image,
        "prompt": prompt,
        "aspect": aspect,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["result"]

def poll_flux_result(request_id, max_wait=120):
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
                'x-key': os.environ.get("BFL_API_KEY"),
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

def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main_image_generation(token,  image_path="", prompt="an image", aspect = "21:9"):
    image_generation_folder = create_app_structure_image_generation()
    if image_path != "":
        image_b64 = encode_image_base64(image_path)
    else: 
        image_b64 = ""
    prompt = get_enhanced_prompt(prompt, token)["enhancedPrompt"]
    print(prompt)
    data = call_flux_pro_ultra(token, image_b64, aspect, prompt)

    request_id = data.get("id") 
    encoded_result = poll_flux_result(request_id)
    path = download_and_save_image_from_url(encoded_result, image_generation_folder)

    return path


