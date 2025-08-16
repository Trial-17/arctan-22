#!/bin/bash

# --- DÉBOGAGE : Redirige toute la sortie vers un fichier log ---
# Pour analyser ce fichier en cas de problème, allez dans /tmp/premiere_copilot_install.log
exec > "/tmp/premiere_copilot_install.log" 2>&1
set -x # Affiche chaque commande avant de l'exécuter

echo "--- DÉBUT DU SCRIPT POSTINSTALL ---"

# --- Étape 1 : Identifier l'utilisateur connecté (et non 'root') ---
LOGGED_IN_USER=$(stat -f%Su /dev/console)
USER_HOME=$(eval echo "~$LOGGED_IN_USER")

# Vérification de sécurité
if [ -z "$USER_HOME" ] || [ "$USER_HOME" == "~" ]; then
    echo "ERREUR : Impossible de déterminer le dossier de l'utilisateur connecté."
    exit 1 # Fait échouer l'installation si l'utilisateur n'est pas trouvé
fi

echo "Utilisateur détecté : $LOGGED_IN_USER"
echo "Dossier utilisateur : $USER_HOME"


# --- Étape 2 : Créer la structure de dossiers au bon endroit ---
echo "📁 Création de la structure API Premiere Copilot..."
BASE_PATH="$USER_HOME/Documents/Adobe/Premiere Pro/Premiere Copilot"
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
done


# --- Étape 3 : Rendre l'API exécutable ---
API_EXECUTABLE='/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/PremiereCopilotAPI/PremiereCopilot'

echo "⚙️  Vérification de l'exécutable de l'API..."
if [ -f "$API_EXECUTABLE" ]; then
    echo "   -> Fichier trouvé. Ajout de la permission d'exécution..."
    chmod +x "$API_EXECUTABLE"
    echo "   ✅ Permission d'exécution ajoutée."
else
    echo "   ⚠️ AVERTISSEMENT : Le fichier API n'a pas été trouvé."
fi


# --- Étape 4 : Corriger les permissions pour le bon utilisateur ---
echo "🔐 Correction des permissions..."
chown -R "$LOGGED_IN_USER" "$BASE_PATH"
chmod -R u+rwX "$BASE_PATH"

echo "✅ Structure API et fichiers prêts."
echo "--- FIN DU SCRIPT POSTINSTALL ---"

exit 0