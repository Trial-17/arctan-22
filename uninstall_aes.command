#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "---------------------------------------------------"
echo "   Désinstallation de Premiere Copilot (AES)       "
echo "---------------------------------------------------"

# 1. Arrêt du processus API (port 8000)
echo "🛑 Arrêt du processus API en cours..."
PIDS=$(lsof -t -i:8000)
if [ -n "$PIDS" ]; then
    kill -9 $PIDS
    echo "   ✅ Processus API arrêté."
else
    echo "   ℹ️  Aucun processus API détecté."
fi

# 2. Suppression de l'extension
EXTENSION_PATH="/Library/Application Support/Adobe/CEP/extensions/PremiereGPTaescripts"
echo "🗑️  Suppression de l'extension : $EXTENSION_PATH"

if [ -d "$EXTENSION_PATH" ]; then
    # Besoin de sudo pour /Library
    if [ -w "$EXTENSION_PATH" ]; then
        rm -rf "$EXTENSION_PATH"
    else
        echo "   🔒 Droits administrateur requis. Veuillez entrer votre mot de passe si demandé."
        sudo rm -rf "$EXTENSION_PATH"
    fi
    
    if [ ! -d "$EXTENSION_PATH" ]; then
        echo "   ✅ Extension supprimée avec succès."
    else
        echo "   ❌ Erreur : Impossible de supprimer l'extension."
    fi
else
    echo "   ℹ️  L'extension n'a pas été trouvée à cet emplacement."
fi

# 3. Message de fin
echo ""
echo "✅ Désinstallation terminée."
echo "Note : Les fichiers utilisateur (cachés, logs, modèles IA téléchargés) dans 'Documents/Adobe/Premiere Pro/Premiere Copilot' n'ont pas été supprimés par sécurité."
echo "Vous pouvez fermer cette fenêtre."
