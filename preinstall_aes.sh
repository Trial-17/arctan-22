#!/bin/bash

# --- Chemins des dossiers à supprimer ---
# Utiliser des variables rend le script plus lisible et facile à modifier.
# Pour la version aescripts, on nettoie tout potentiel dossier existant
FOLDER_PATH_1="/Library/Application Support/Adobe/CEP/extensions/PremiereCopilot"
FOLDER_PATH_2="/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta"
FOLDER_PATH_3="/Library/Application Support/Adobe/CEP/extensions/PremiereGPTaescripts"

echo "Lancement du script de pré-installation (Version aescripts)..."

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
    echo "   -> Dossier non trouvé (PremiereGPTaescripts). Aucune action."
fi

# ⚠️ Le script DOIT se terminer par "exit 0" pour que l'installation continue.
# S'il se termine avec un autre code, l'installateur .pkg s'arrêtera.
exit 0
