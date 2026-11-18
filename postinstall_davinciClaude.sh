#!/bin/bash

echo "📁 Création de la structure API davinciClaude..."

# NB : dossier de données conservé en "DaVinciGPT" — le bundle servi par l'API
# écrit en dur dans ~/Documents/DaVinciGPT (getDocumentsBase). Ne pas renommer
# tant que le bundle DaVinci dédié n'a pas changé ce chemin.
# Le script tourne en root : on vise l'utilisateur connecté à la session, pas $HOME (root).
CONSOLE_USER=$(stat -f%Su /dev/console)
if [ -z "$CONSOLE_USER" ] || [ "$CONSOLE_USER" = "root" ]; then CONSOLE_USER="$USER"; fi
USER_HOME=$(dscl . -read "/Users/$CONSOLE_USER" NFSHomeDirectory 2>/dev/null | awk '{print $2}')
[ -z "$USER_HOME" ] && USER_HOME="$HOME"
BASE_PATH="$USER_HOME/Documents/DaVinciGPT"
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
chown -R "$CONSOLE_USER" "$BASE_PATH"
chmod -R u+rwX "$BASE_PATH"

echo "✅ Structure et fichiers prêts dans : $BASE_PATH"

exit 0
