#!/bin/bash

# --- Chemins des dossiers à supprimer ---
# macOS : chemin SANS "Support" (cf. doc Resolve Workflow Integration).
PLUGINS_DIR="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins"
# Réinstallation propre du nouveau plugin + suppression de l'ancien DavinciGPT (migration).
FOLDER_PATH_1="$PLUGINS_DIR/davinciClaude"
FOLDER_PATH_2="$PLUGINS_DIR/DavinciGPT"

echo "Lancement du script de pré-installation davinciClaude..."

for FOLDER_PATH in "$FOLDER_PATH_1" "$FOLDER_PATH_2"; do
    if [ -d "$FOLDER_PATH" ]; then
        echo "🗑️  Ancienne version trouvée. Suppression de : $FOLDER_PATH"
        rm -rf "$FOLDER_PATH"
        echo "   ✅ Suppression terminée."
    else
        echo "   -> Dossier non trouvé : $FOLDER_PATH. Aucune action."
    fi
done

exit 0
