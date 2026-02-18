#!/bin/bash

# --- Chemins des dossiers à supprimer ---
# Utiliser des variables rend le script plus lisible et facile à modifier.
# Pour la version aescript, on nettoie tout potentiel dossier existant
FOLDER_PATH_1="/Library/Application Support/Adobe/CEP/extensions/PremiereCopilot"
FOLDER_PATH_2="/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta"
FOLDER_PATH_3="/Library/Application Support/Adobe/CEP/extensions/PremiereGPTaescript"

echo "Lancement du script de pré-installation (Version AESCRIPT)..."

# --- Arrêt forcé de l'API précédente (port 8000) ---
echo "🔌 Arrêt forcé des processus sur le port 8000..."
# Force l'arrêt de tous les processus utilisant le port 8000
# || true empêche l'échec du script si aucun processus n'est trouvé
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
echo "   ✅ Nettoyage du port 8000 terminé."

# --- Traitement des anciens dossiers ---
# L'option -d vérifie si le chemin existe ET est un dossier.
if [ -d "$FOLDER_PATH_1" ]; then
    echo "🗑️  Ancienne version trouvée. Suppression de : $FOLDER_PATH_1"
    # rm -rf supprime le dossier et tout son contenu sans demander de confirmation.
    rm -rf "$FOLDER_PATH_1"
    echo "   ✅ Suppression terminée."
else
    echo "   -> Dossier non trouvé (PremiereCopilot). Aucune action."
fi

if [ -d "$FOLDER_PATH_2" ]; then
    echo "🗑️  Ancienne version trouvée. Suppression de : $FOLDER_PATH_2"
    rm -rf "$FOLDER_PATH_2"
    echo "   ✅ Suppression terminée."
else
    echo "   -> Dossier non trouvé (PremiereGPTBeta). Aucune action."
fi

if [ -d "$FOLDER_PATH_3" ]; then
    echo "🗑️  Ancienne version trouvée. Suppression de : $FOLDER_PATH_3"
    rm -rf "$FOLDER_PATH_3"
    echo "   ✅ Suppression terminée."
else
    echo "   -> Dossier non trouvé (PremiereGPTaescript). Aucune action."
fi

# ⚠️ Le script DOIT se terminer par "exit 0" pour que l'installation continue.
# S'il se termine avec un autre code, l'installateur .pkg s'arrêtera.
exit 0
