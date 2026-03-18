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

# --- Rendre l'API exécutable ---
API_EXECUTABLE='/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins/DavinciGPT/js/libs/PremiereCopilotAPI/PremiereCopilot'

echo "⚙️  Vérification de l'exécutable de l'API..."
if [ -f "$API_EXECUTABLE" ]; then
    echo "   -> Fichier trouvé. Ajout de la permission d'exécution..."
    chmod +x "$API_EXECUTABLE"
    echo "   ✅ Permission d'exécution ajoutée."
else
    echo "   ⚠️ AVERTISSEMENT : Le fichier API n'a pas été trouvé à l'emplacement attendu."
fi

# Fix des permissions
echo "🔐 Correction des permissions..."
chown -R "$USER" "$BASE_PATH"
chmod -R u+rwX "$BASE_PATH"

echo "✅ Structure API et fichiers prêts dans : $BASE_PATH"

echo "🚀 Lancement de l'API..."
sudo -u "$USER" nohup "$API_EXECUTABLE" > /dev/null 2>&1 &
echo "   ✅ API lancée."

exit 0
