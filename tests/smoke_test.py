import subprocess
import time
import requests
import sys
import os

def smoke_test():
    # 1. Chemin vers l'exécutable construit par PyInstaller
    executable = "./dist/PremiereCopilot/PremiereCopilot"
    
    if not os.path.exists(executable):
        print(f"❌ Erreur : Exécutable non trouvé à {executable}")
        sys.exit(1)

    print(f"🚀 Lancement de l'API : {executable}...")
    
    # 2. Lancer le processus en arrière-plan
    process = subprocess.Popen(
        [executable],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 3. Attendre que l'API soit prête (retry loop)
    max_retries = 180
    api_ready = False
    
    print("⏳ Attente du démarrage de l'API...")
    for i in range(max_retries):
        # Vérification immédiate : le processus a-t-il crashé ?
        if process.poll() is not None:
            print("❌ Le processus de l'API s'est arrêté prématurément !")
            break
            
        time.sleep(1)
        try:
            # On teste l'URL racine de ton FastAPI
            response = requests.get("http://127.0.0.1:8000/", timeout=2)
            if response.status_code in [200, 404]: # 404 est OK aussi si tu n'as pas de route racine
                print("✅ L'API répond correctement !")
                api_ready = True
                break
        except requests.exceptions.ConnectionError:
            print(f"   (Essai {i+1}/{max_retries}) L'API ne répond pas encore...")

    # 4. Nettoyage
    print("🛑 Arrêt du processus...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    if api_ready:
        print("\n✨ SMOKE TEST RÉUSSI : L'exécutable démarre et répond.")
        sys.exit(0)
    else:
        print("\n❌ SMOKE TEST ÉCHOUÉ : L'API n'a jamais répondu.")
        # Afficher les erreurs s'il y en a
        stdout, stderr = process.communicate()
        print("--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        sys.exit(1)

if __name__ == "__main__":
    smoke_test()
