#!/bin/bash

echo "📁 Création de la structure API Premiere Copilot (Version aescripts)..."

BASE_PATH="$HOME/Documents/Adobe/Premiere Pro/Premiere Copilot"
mkdir -p "$BASE_PATH"

# Liste des dossiers à créer
for folder in \
    "audio_sync" \
    "image_generation" \
    "matplotlib_cache" \
    "music_analysis" \
    "rush_db" \
    "script" \
    "seq_preset" \
    "sfx" \
    "temp" \
    "thumbnails" \
    "transcription_analysis"
do
    mkdir -p "$BASE_PATH/$folder"
    echo "📂 Dossier créé : $BASE_PATH/$folder"
done

# --- Rendre l'API exécutable ---
# CHEMIN CIBLE POUR LA VERSION aescripts : PremiereGPTaescripts
API_EXECUTABLE='/Library/Application Support/Adobe/CEP/extensions/PremiereGPTaescripts/js/libs/PremiereCopilotAPI/PremiereCopilot'

echo "⚙️  Vérification de l'exécutable de l'API..."
if [ -f "$API_EXECUTABLE" ]; then
    echo "   -> Fichier trouvé. Ajout de la permission d'exécution..."
    chmod +x "$API_EXECUTABLE"
    echo "   ✅ Permission d'exécution ajoutée."
else
    echo "   ⚠️ AVERTISSEMENT : Le fichier API n'a pas été trouvé à l'emplacement attendu."
    echo "   Attendu : $API_EXECUTABLE"
fi

# Fix des permissions
echo "🔐 Correction des permissions..."
chown -R "$USER" "$BASE_PATH"
chmod -R u+rwX "$BASE_PATH"

echo "✅ Structure API et fichiers prêts dans : $BASE_PATH"
exit 0
