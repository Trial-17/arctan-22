#!/bin/bash

echo "📁 Création de la structure API DaVinciGPT..."

BASE_PATH="$HOME/Documents/DaVinciGPT"
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

# Fix des permissions
echo "🔐 Correction des permissions..."
chown -R "$USER" "$BASE_PATH"
chmod -R u+rwX "$BASE_PATH"

echo "✅ Structure et fichiers prêts dans : $BASE_PATH"

exit 0
