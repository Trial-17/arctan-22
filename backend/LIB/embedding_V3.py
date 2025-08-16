# import json
# import pickle

# import numpy as np
# import copy
# from LIB import config
# from pathlib import Path
# import os


# import en_core_web_lg 
# nlp = en_core_web_lg.load()

# # model = spacy.load("en_core_web_lg")
# # print(model._path) 

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

# GLOBAL_DB_PATH = str(create_app_structure()['rush_db'] + "/rush_db.json")
# VECTOR_DB_PATH = str(create_app_structure()['rush_db'] + "/rush_db.pkl")

    
# def sync_vectorized_rush_db():
#     # Paths
#     config.API_STATUS = "Embedding"
#     # Charger la DB JSON brute
#     with open(GLOBAL_DB_PATH, "r") as f:
#         raw_db = json.load(f)

#     # Charger ou initialiser la base vectorisée
#     if Path(VECTOR_DB_PATH).exists():
#         with open(VECTOR_DB_PATH, "rb") as f:
#             vectorized_db = pickle.load(f)
#     else:
#         vectorized_db = []

#     # Indexer les fichiers déjà vectorisés
#     existing_files = {r['file'] for r in vectorized_db}

#     # Vectoriser uniquement les rushs qui n'existent pas encore
    
#     new_rushes = []
#     for rush in raw_db:
#         if rush['file'] not in existing_files:
#             config.API_STATUS = "Vectorisation"
#             vector_cache = []
#             for field in rush["data"].values():
#                 if isinstance(field, list):
#                     for val in field:
#                         vector_cache.append((val, nlp(val).vector))
#                 elif isinstance(field, str):
#                     vector_cache.append((field, nlp(field).vector))
#             rush["vector_cache"] = vector_cache
#             new_rushes.append(rush)

#     # Fusion et sauvegarde
#     updated_db = vectorized_db + new_rushes
#     with open(VECTOR_DB_PATH, "wb") as f:
#         pickle.dump(updated_db, f)

#     # print(f"✅ Base vectorisée synchronisée. {len(new_rushes)} nouveau(x) rush(s) ajouté(s).")


# def load_vectorized_db():
#     # 1. Vérifier si rush_db.json existe, sinon le créer vide
#     if not Path(GLOBAL_DB_PATH).exists():
#         with open(GLOBAL_DB_PATH, "w", encoding="utf-8") as f:
#             json.dump([], f)
#         print("⚠️ rush_db.json créé car il était manquant.")

#     # 2. Vérifier si rush_db.pkl existe, sinon lancer la vectorisation
#     if not Path(VECTOR_DB_PATH).exists():
#         print("⚠️ rush_db.pkl manquant. Lancement de la vectorisation...")
#         sync_vectorized_rush_db()

#     # 3. Charger la base vectorisée
#     with open(VECTOR_DB_PATH, "rb") as f:
#         return pickle.load(f)

# def cosine_sim(vec1, vec2):
#     if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
#         return 0.0
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def search_best_matches_vectorized(db, query, top_k=10, threshold=0.55):
#     tokens = query.lower().split()
#     token_vectors = [nlp(token).vector for token in tokens]

#     scored_results = []
#     for rush in db:
#         score = 0
#         for token_vec in token_vectors:
#             for _, field_vec in rush["vector_cache"]:
#                 sim = cosine_sim(token_vec, field_vec)
#                 if sim > threshold:
#                     score += sim
#         if score > 0:
#             scored_results.append((rush, score))

#     scored_results.sort(key=lambda x: x[1], reverse=True)

#     # Supprime proprement vector_cache de chaque rush avant retour
#     cleaned_results = []
#     for rush, _ in scored_results[:top_k]:
#         rush_copy = copy.deepcopy(rush)
#         if "vector_cache" in rush_copy:
#             del rush_copy["vector_cache"]
#         cleaned_results.append(rush_copy)

#     return cleaned_results



