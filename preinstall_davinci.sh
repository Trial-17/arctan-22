#!/bin/bash

# --- Chemins des dossiers à supprimer ---
FOLDER_PATH_1="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Workflow Integration Plugins/DavinciGPT"

echo "Lancement du script de pré-installation DaVinci..."

# --- Arrêt forcé de l'API précédente (port 8000) ---
echo "🔌 Arrêt forcé des processus sur le port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
echo "   ✅ Nettoyage du port 8000 terminé."

# --- Traitement du premier dossier ---
if [ -d "$FOLDER_PATH_1" ]; then
    echo "🗑️  Ancienne version trouvée. Suppression de : $FOLDER_PATH_1"
    rm -rf "$FOLDER_PATH_1"
    echo "   ✅ Suppression terminée."
else
    echo "   -> Dossier non trouvé. Aucune action."
fi

exit 0
