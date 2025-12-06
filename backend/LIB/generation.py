import os
from PIL import Image
from pathlib import Path
from datetime import datetime
import requests
from LIB import config
import json
import wave
import struct



def get_enhanced_prompt(prompt, token, output_type="image"):
    """
    Appelle l'API externe pour améliorer le prompt de génération d'image ou de vidéo.
    """
    try:
        response = requests.post(
            f"{config.API_URL}/enhance-prompt",
            json={"prompt": prompt, "outputType": output_type},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Erreur lors de l'amélioration du prompt: {response.status_code}")
            return {"enhancedPrompt": prompt}  # En cas d'erreur, retourne le prompt original
            
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API d'amélioration de prompt: {str(e)}")
        return {"enhancedPrompt": prompt}  # En cas d'erreur, retourne le prompt original

def creer_masque_transparence(chemin_image_png: str, model: str):
    """
    Crée un masque noir et blanc à partir de la transparence d'une image PNG.

    Cette fonction ouvre une image PNG, identifie ses zones transparentes et
    génère une nouvelle image de masque. Sur ce masque, les pixels opaques ou
    semi-transparents de l'original sont blancs, et les pixels entièrement
    transparents sont noirs.

    Le masque est sauvegardé dans le même répertoire que l'image originale,
    avec le suffixe '_mask' ajouté au nom du fichier.

    Args:
        chemin_image_png (str): Le chemin d'accès complet au fichier PNG source.
    """
    # Vérification que le fichier existe
    if not os.path.isfile(chemin_image_png):
        print(f"Erreur : Le fichier '{chemin_image_png}' n'existe pas.")
        return

    try:
        # Ouvrir l'image et la convertir en RGBA pour garantir un canal alpha
        img = Image.open(chemin_image_png).convert('RGBA')

        # Extraire le canal alpha (la transparence)
        # C'est une image en niveaux de gris : le noir est transparent, le blanc est opaque.
        alpha = img.getchannel('A')

        # Créer le masque binaire.
        # La fonction point() applique une opération à chaque pixel.
        # Si la valeur du pixel alpha (p) est > 0 (opaque), on le met en blanc (255).
        # Sinon (transparent), on le met en noir (0).
        # Le mode 'L' correspond à une image 8-bit en niveaux de gris.
        if model.startswith("black-forest-labs/flux-fill"):
            masque = alpha.point(lambda p: 255 if p == 0 else 0, mode='L')
        else:
            masque = alpha.point(lambda p: 255 if p > 0 else 0, mode='L')

        # Définir le chemin de sauvegarde pour le masque
        # On sépare le nom du fichier et son extension
        base, ext = os.path.splitext(chemin_image_png)
        chemin_masque = f"{base}_mask.png"

        # Sauvegarder la nouvelle image (le masque)
        masque.save(chemin_masque)
        return chemin_masque


    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")

def create_app_structure_image_generation():
    """
    Crée la structure de dossiers de l'application.
    """
    base_path = Path.home() / "Documents" / "Adobe" / "Premiere Pro" / "Premiere Copilot"
    image_generation_path = base_path / "image_generation"
    image_generation_path.mkdir(parents=True, exist_ok=True)
    return image_generation_path

def validate_wav_file(file_path: str) -> bool:
    """
    Valide qu'un fichier WAV est correctement formaté et lisible.
    """
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # Vérifier les paramètres de base
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # Vérifier que le fichier a du contenu
            if frames == 0:
                print("❌ Fichier WAV vide (0 frames)")
                return False
                
            # Vérifier que la durée est raisonnable (pas infinie)
            duration = frames / sample_rate
            if duration > 3600:  # Plus d'1 heure
                print(f"⚠️ Durée suspecte: {duration:.2f} secondes")
                
            print(f"✅ WAV valide: {duration:.2f}s, {sample_rate}Hz, {channels} canal(s), {sample_width} bytes/sample")
            return True
            
    except Exception as e:
        print(f"❌ Erreur de validation WAV: {e}")
        return False

def normalize_wav_audio(file_path: str) -> str:
    """
    Normalise et corrige les données audio d'un fichier WAV pour assurer la compatibilité.
    """
    try:
        with wave.open(file_path, 'rb') as wav_in:
            # Lire tous les paramètres
            frames = wav_in.getnframes()
            sample_rate = wav_in.getframerate()
            channels = wav_in.getnchannels()
            sample_width = wav_in.getsampwidth()
            
            print(f"🔍 Paramètres originaux: {sample_rate}Hz, {channels} canal(s), {sample_width} bytes/sample, {frames} frames")
            
            # Lire toutes les données audio
            raw_audio = wav_in.readframes(frames)
            
            # Vérifier si les données sont vides ou corrompues
            if len(raw_audio) == 0:
                print("❌ Aucune donnée audio trouvée")
                return file_path
            
            # Normaliser les données audio
            if sample_width == 2:  # 16-bit
                # Convertir en entiers signés 16-bit
                audio_data = struct.unpack(f'<{len(raw_audio)//2}h', raw_audio)
                
                # Garder le stéréo si c'est déjà stéréo, sinon rester mono
                if channels == 2:
                    print("🔄 Conservation du stéréo...")
                    # Pas de conversion, on garde les données stéréo telles quelles
                    print(f"✅ Stéréo conservé: {len(audio_data)} échantillons stéréo")
                
                # Normaliser l'amplitude (éviter la saturation)
                max_val = max(abs(x) for x in audio_data) if audio_data else 1
                if max_val > 0:
                    normalized_data = [int(x * 0.8 * 32767 / max_val) for x in audio_data]
                else:
                    normalized_data = audio_data
                
                # Reconvertir en bytes
                normalized_bytes = struct.pack(f'<{len(normalized_data)}h', *normalized_data)
            else:
                # Pour d'autres formats, garder les données telles quelles
                normalized_bytes = raw_audio
        
        # Créer un nouveau fichier WAV normalisé
        normalized_filename = file_path.replace('.wav', '_normalized.wav')
        
        with wave.open(normalized_filename, 'wb') as wav_out:
            # Paramètres optimisés pour la compatibilité - PRÉSERVER la fréquence originale
            wav_out.setnchannels(channels)  # Conserver le nombre de canaux original (stéréo ou mono)
            wav_out.setsampwidth(2)  # 16-bit
            wav_out.setframerate(sample_rate)  # Conserver la fréquence d'échantillonnage originale
            wav_out.writeframes(normalized_bytes)
        
        print(f"✅ WAV normalisé sauvegardé (fréquence préservée: {sample_rate}Hz, {channels} canal(s)): {normalized_filename}")
        return normalized_filename
        
    except Exception as e:
        print(f"❌ Erreur lors de la normalisation WAV: {e}")
        return file_path

def fix_wav_header(file_path: str) -> str:
    """
    Corrige l'en-tête WAV si nécessaire pour assurer la compatibilité.
    """
    try:
        # Lire le fichier
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Vérifier si c'est un WAV valide
        if not data.startswith(b'RIFF') or b'WAVE' not in data[:12]:
            print("❌ Fichier ne semble pas être un WAV valide")
            return file_path
        
        # Créer un nouveau fichier WAV correctement formaté
        fixed_filename = file_path.replace('.wav', '_fixed.wav')
        
        # Extraire les données audio (après l'en-tête)
        with wave.open(fixed_filename, 'wb') as wav_out:
            # Paramètres par défaut pour la compatibilité
            wav_out.setnchannels(1)  # Mono
            wav_out.setsampwidth(2)  # 16-bit
            wav_out.setframerate(44100)  # 44.1 kHz
            
            # Trouver les données audio dans le fichier original
            try:
                with wave.open(file_path, 'rb') as wav_in:
                    frames = wav_in.readframes(wav_in.getnframes())
                    wav_out.writeframes(frames)
            except:
                # Si on ne peut pas lire le WAV original, essayer d'extraire les données brutes
                # Chercher le début des données audio (après 'data')
                data_start = data.find(b'data') + 8
                if data_start > 7:
                    audio_data = data[data_start:]
                    wav_out.writeframes(audio_data)
        
        print(f"✅ WAV corrigé sauvegardé: {fixed_filename}")
        return fixed_filename
        
    except Exception as e:
        print(f"❌ Erreur lors de la correction WAV: {e}")
        return file_path

def download_demucs_file(url: str, stem_name: str) -> str:
    """
    Télécharge un fichier Demucs de manière simple sans traitement WAV complexe.
    """
    output_folder = create_app_structure_image_generation()
    
    # Nom de fichier avec le nom du stem
    filename = f"demucs_{stem_name}_{datetime.now().strftime('%y%m%d%H%M%S')}.wav"
    output_path = output_folder / filename
    
    try:
        print(f"🎵 Téléchargement simple du stem {stem_name}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Sauvegarde directe sans traitement
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        print(f"✅ Stem {stem_name} téléchargé directement: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Erreur téléchargement stem {stem_name}: {e}")
        return None

def download_and_save_from_url(url: str, force_extension: str = None): # Renamed output to url for clarity
    output_folder = create_app_structure_image_generation()
    
    # Your filename logic is fine
    if force_extension:
        extension = force_extension
    else:
        extension = url.split('.')[-1]
        if extension not in ["png", "jpg", "jpeg", "mp4", "wav", "mp3", "webp"]:
            extension = "png" # Default extension if not found
    
    filename = datetime.now().strftime(f"gen_%y%m%d%H%M%S.{extension}")
    output_path = output_folder / filename

    try:
        # 2. Make a GET request to download the data from the URL
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error if the download fails (e.g., 404)

        # 3. Write the downloaded content (in bytes) to the file
        with open(output_path, "wb") as f:
            f.write(response.content) # <-- Use response.content instead of output.read()
        
        # 4. Traitement spécial selon le type de fichier
        if extension.lower() in ["webp", "jpg", "jpeg"]:
            try:
                # Ouvrir l'image avec Pillow
                with Image.open(output_path) as img:
                    # Convertir en RGBA pour préserver la transparence (ou RGB si pas de transparence)
                    if img.mode in ('RGBA', 'LA'):
                        img = img.convert('RGBA')
                    else:
                        img = img.convert('RGB')
                    
                    # Créer un nouveau nom de fichier avec l'extension PNG
                    png_filename = datetime.now().strftime(f"gen_%y%m%d%H%M%S.png")
                    png_output_path = output_folder / png_filename
                    
                    # Sauvegarder en PNG
                    img.save(png_output_path, 'PNG')
                    
                    # Supprimer le fichier original
                    os.remove(output_path)
                    
                    print(f"{extension.upper()} file converted to PNG and saved to {png_output_path}")
                    return str(png_output_path)
                    
            except Exception as e:
                print(f"Error converting {extension.upper()} to PNG: {e}")
                # En cas d'erreur de conversion, retourner le fichier original
                print(f"File saved as {extension.upper()} to {output_path}")
                return str(output_path)
                
        elif extension.lower() == "wav":
            # Validation et traitement du fichier WAV
            print("🔍 Validation du fichier WAV...")
            if not validate_wav_file(str(output_path)):
                print("🔧 Tentative de correction du fichier WAV...")
                fixed_path = fix_wav_header(str(output_path))
                if fixed_path != str(output_path):
                    # Supprimer le fichier original et utiliser le fichier corrigé
                    os.remove(output_path)
                    output_path = Path(fixed_path)
                    print(f"✅ Fichier WAV corrigé et sauvegardé: {output_path}")
                else:
                    print(f"⚠️ Impossible de corriger le WAV, fichier original conservé: {output_path}")
            else:
                print(f"✅ Fichier WAV valide détecté")
            
            # Normaliser tous les fichiers WAV pour assurer la compatibilité maximale
            print("🔧 Normalisation des données audio pour la compatibilité...")
            normalized_path = normalize_wav_audio(str(output_path))
            if normalized_path != str(output_path):
                # Supprimer le fichier original et utiliser le fichier normalisé
                os.remove(output_path)
                output_path = Path(normalized_path)
                print(f"✅ Fichier WAV normalisé et sauvegardé: {output_path}")
            else:
                print(f"⚠️ Normalisation échouée, fichier original conservé: {output_path}")
            
            return str(output_path)
        else:
            print(f"File successfully saved to {output_path}")
            return str(output_path)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return None

def main_generation(input_data, token):
    
    if 'model' not in input_data:
        raise ValueError("Le modèle n'est pas spécifié dans les données d'entrée")
    
    model = input_data.pop('model')
    
    # Créer un masque si image est présent
    if 'image' in input_data:
        input_data['mask'] = creer_masque_transparence(input_data['image'], model)

    url = f"{config.API_URL}/generation"
    headers = {
        'Authorization': f'Bearer {token}',
    }

    files_to_send = {}
    input_json_data = input_data.copy()
    open_files = []

    try:
        # Parcourir tous les champs de input_data
        for key, value in input_data.items():
            if isinstance(value, list):
                # Si c'est une liste, parcourir ses éléments
                file_count = 0
                for i, item in enumerate(value):
                    if isinstance(item, str) and os.path.exists(item):
                        print(f"Preparing {key} file for upload: '{item}' (index {i})")
                        fh = open(item, 'rb')
                        # Envoyer chaque fichier individuellement avec un nom de clé unique
                        files_to_send[f"{key}_{i}"] = (os.path.basename(item), fh)
                        input_json_data[key][i] = f"{key}_{i}"
                        open_files.append(fh)
                        file_count += 1
                    else:
                        print(f"Warning: {key} file path not found, skipping: {item}")
                
                # Remplacer la liste par "add_file" si des fichiers ont été trouvés
                if file_count == 0:
                     input_json_data[key] = []

                    
            elif isinstance(value, str) and os.path.exists(value):
                # Si c'est un string et que c'est un chemin de fichier valide
                print(f"Preparing file for upload: '{value}' for key '{key}'")
                fh = open(value, 'rb')
                files_to_send[key] = (os.path.basename(value), fh)
                open_files.append(fh)
                # Remplacer le chemin par "add_file"
                input_json_data[key] = str(key)
            else:
                # Garder la valeur originale
                input_json_data[key] = value

        payload = {
            "model": model,
            "input_json": json.dumps(input_json_data)
        }

        print(f"Sending request to API. Payload: {payload}, Files: {list(files_to_send.keys())}")
        response = requests.post(url, headers=headers, data=payload, files=files_to_send)
        print(f"API Response Status Code: {response.status_code}")
        
        if response.status_code == 403:
            # Erreur d'abonnement ou de crédits
            try:
                error_detail = response.json().get('detail', 'Inactive or insufficient subscription')
            except:
                error_detail = 'Inactive or insufficient subscription'
            raise PermissionError(f"subscription_error: {error_detail}")
        elif response.status_code != 200:
            try:
                error_detail = response.json().get('detail', response.text)
            except:
                error_detail = response.text
            raise Exception(f"Erreur API (Status {response.status_code}): {error_detail}")

        result_json = response.json()
        if "result" not in result_json:
            raise Exception("La réponse de l'API ne contient pas de résultat")
            
        output = result_json["result"]
        print(output)
        
        if not output:
            raise Exception("L'API a retourné un résultat vide")
        
        # Gestion spéciale pour Demucs qui retourne un dictionnaire de fichiers
        if model.startswith("ryan5453/demucs") and isinstance(output, dict):
            print("🎵 Traitement des stems audio Demucs...")
            downloaded_paths = {}
            for stem_name, url in output.items():
                # Utiliser la fonction simple pour Demucs (sans traitement WAV complexe)
                downloaded_path = download_demucs_file(url, stem_name)
                if downloaded_path is None:
                    print(f"⚠️ Échec du téléchargement du stem {stem_name}")
                    continue
                downloaded_paths[stem_name] = downloaded_path
            
            if not downloaded_paths:
                raise Exception("Échec du téléchargement de tous les stems")
            
            return {"result": downloaded_paths, "type": "demucs"}
        
        
        # Gestion spéciale pour le clonage de voix Minimax
        elif model.startswith("minimax/voice-cloning"):
            print("🎤 Traitement du clonage de voix Minimax...")
            if isinstance(output, dict) and 'voice_id' in output:
                voice_id = output['voice_id']
                print(f"🎤 Voice ID: {voice_id}")
                
                # Afficher l'alerte avec l'ID de voix
                print(f"⚠️ IMPORTANT: Voice ID is {voice_id} - Copy this ID as it will only be shown once!")
                
                return {"result": voice_id, "type": "voice_cloning"}
            elif isinstance(output, str):
                # L'API retourne directement l'ID de voix comme string
                voice_id = output
                print(f"🎤 Voice ID: {voice_id}")
                
                # Afficher l'alerte avec l'ID de voix
                print(f"⚠️ IMPORTANT: Voice ID is {voice_id} - Copy this ID as it will only be shown once!")
                
                return {"result": voice_id, "type": "voice_cloning"}
            else:
                raise Exception("Format de sortie invalide pour le clonage de voix")
        else:
            # Gestion standard pour les autres modèles
            downloaded_path = download_and_save_from_url(output)
            
            if downloaded_path is None:
                raise Exception("Échec du téléchargement du fichier généré")
            
            return downloaded_path
        
    except requests.exceptions.Timeout:
        raise Exception("La requête a expiré (timeout). La génération prend trop de temps.")
    except requests.exceptions.ConnectionError:
        raise Exception("Impossible de se connecter à l'API de génération. Vérifiez votre connexion.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Erreur de requête: {str(e)}")
    finally:
        for f in open_files:
            f.close()
