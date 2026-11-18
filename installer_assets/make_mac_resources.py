"""Génère les pages HTML de l'installeur macOS (accueil + fin), une par langue.

Usage : python3 installer_assets/make_mac_resources.py
Sorties : installer_assets/<marque>/mac/<langue>.lproj/{welcome,conclusion}.html
Installer.app choisit le dossier .lproj selon la langue du Mac (anglais par défaut).
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Libellés des menus de l'application hôte, tels qu'affichés dans chaque langue.
# Resolve : menus laissés en anglais (interface par défaut).
PREMIERE_MENU = {
    "en": "Window > Extensions > Premiere GPT",
    "fr": "Fenêtre > Extensions > Premiere GPT",
    "de": "Fenster > Erweiterungen > Premiere GPT",
    "es": "Ventana > Extensiones > Premiere GPT",
    "pt": "Janela > Extensões > Premiere GPT",
    "ja": "ウィンドウ > エクステンション > Premiere GPT",
    "ko": "창 > 확장 > Premiere GPT",
    "hi": "Window > Extensions > Premiere GPT",
}
DAVINCI_MENU = "Workspace > Workflow Integrations > davinciClaude"

BRANDS = {
    "premiere": {"name": "PremiereCopilot", "host": "Premiere Pro", "menu": PREMIERE_MENU},
    "davinci": {"name": "DavinciClaude", "host": "DaVinci Resolve", "menu": {k: DAVINCI_MENU for k in PREMIERE_MENU}},
}

T = {
    "en": {
        "welcome_title": "Welcome to {name}",
        "welcome_body": "This will install {name} on your Mac. It only takes a few seconds.",
        "close_host": "Please close {host} before continuing.",
        "done_title": "{name} is ready",
        "how": "To open it:",
        "step1": "Launch {host} (restart it if it was open).",
        "step2": "Go to <b>{menu}</b>.",
        "step3": "Sign in, and the guided tour takes it from there.",
        "help": "Need help? Read the <a href=\"{docs}\">getting started guide</a>.",
    },
    "fr": {
        "welcome_title": "Bienvenue dans {name}",
        "welcome_body": "{name} va être installé sur votre Mac. Cela ne prend que quelques secondes.",
        "close_host": "Fermez {host} avant de continuer.",
        "done_title": "{name} est prêt",
        "how": "Pour l'ouvrir :",
        "step1": "Lancez {host} (redémarrez-le s'il était ouvert).",
        "step2": "Allez dans <b>{menu}</b>.",
        "step3": "Connectez-vous, la visite guidée prend le relais.",
        "help": "Besoin d'aide ? Consultez le <a href=\"{docs}\">guide de démarrage</a>.",
    },
    "de": {
        "welcome_title": "Willkommen bei {name}",
        "welcome_body": "{name} wird jetzt auf Ihrem Mac installiert. Das dauert nur wenige Sekunden.",
        "close_host": "Bitte schließen Sie {host}, bevor Sie fortfahren.",
        "done_title": "{name} ist bereit",
        "how": "So öffnen Sie es:",
        "step1": "Starten Sie {host} (neu starten, falls es geöffnet war).",
        "step2": "Öffnen Sie <b>{menu}</b>.",
        "step3": "Melden Sie sich an, die geführte Tour übernimmt den Rest.",
        "help": "Brauchen Sie Hilfe? Lesen Sie die <a href=\"{docs}\">Einstiegsanleitung</a>.",
    },
    "es": {
        "welcome_title": "Te damos la bienvenida a {name}",
        "welcome_body": "Se instalará {name} en tu Mac. Solo tarda unos segundos.",
        "close_host": "Cierra {host} antes de continuar.",
        "done_title": "{name} está listo",
        "how": "Para abrirlo:",
        "step1": "Abre {host} (reinícialo si estaba abierto).",
        "step2": "Ve a <b>{menu}</b>.",
        "step3": "Inicia sesión y el tour guiado se encarga del resto.",
        "help": "¿Necesitas ayuda? Lee la <a href=\"{docs}\">guía de inicio</a>.",
    },
    "pt": {
        "welcome_title": "Boas-vindas ao {name}",
        "welcome_body": "O {name} será instalado no seu Mac. Leva só alguns segundos.",
        "close_host": "Feche o {host} antes de continuar.",
        "done_title": "O {name} está pronto",
        "how": "Para abrir:",
        "step1": "Abra o {host} (reinicie se ele estava aberto).",
        "step2": "Vá em <b>{menu}</b>.",
        "step3": "Faça login e o tour guiado cuida do resto.",
        "help": "Precisa de ajuda? Leia o <a href=\"{docs}\">guia de introdução</a>.",
    },
    "ja": {
        "welcome_title": "{name} へようこそ",
        "welcome_body": "{name} をこの Mac にインストールします。数秒で完了します。",
        "close_host": "続行する前に {host} を終了してください。",
        "done_title": "{name} の準備ができました",
        "how": "開き方:",
        "step1": "{host} を起動します（起動中の場合は再起動）。",
        "step2": "<b>{menu}</b> を選びます。",
        "step3": "サインインすると、ガイドツアーが始まります。",
        "help": "お困りですか？<a href=\"{docs}\">はじめにガイド</a>をご覧ください。",
    },
    "ko": {
        "welcome_title": "{name}에 오신 것을 환영합니다",
        "welcome_body": "{name}을(를) 이 Mac에 설치합니다. 몇 초면 끝납니다.",
        "close_host": "계속하기 전에 {host}을(를) 종료하세요.",
        "done_title": "{name} 준비 완료",
        "how": "여는 방법:",
        "step1": "{host}을(를) 실행합니다(열려 있었다면 다시 시작).",
        "step2": "<b>{menu}</b>를 선택합니다.",
        "step3": "로그인하면 가이드 투어가 시작됩니다.",
        "help": "도움이 필요하신가요? <a href=\"{docs}\">시작 가이드</a>를 확인하세요.",
    },
    "hi": {
        "welcome_title": "{name} में आपका स्वागत है",
        "welcome_body": "{name} आपके Mac पर इंस्टॉल होगा। इसमें बस कुछ सेकंड लगेंगे।",
        "close_host": "आगे बढ़ने से पहले {host} बंद करें।",
        "done_title": "{name} तैयार है",
        "how": "इसे खोलने के लिए:",
        "step1": "{host} खोलें (अगर पहले से खुला था तो उसे फिर से शुरू करें)।",
        "step2": "<b>{menu}</b> पर जाएँ।",
        "step3": "साइन इन करें, गाइडेड टूर बाकी काम कर देगा।",
        "help": "मदद चाहिए? <a href=\"{docs}\">शुरुआती गाइड</a> पढ़ें।",
    },
}

DOCS = "https://docs.premierecopilot.com/getting-started/introduction"

CSS = """
:root { color-scheme: light dark; }
body { font: 13px/1.55 -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; margin: 6px 4px; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 12px; letter-spacing: -0.01em; }
p { margin: 0 0 10px; }
.mute { opacity: 0.7; }
ol { margin: 4px 0 14px; padding-left: 20px; }
li { margin: 0 0 6px; }
b { font-weight: 600; }
a { color: #00a866; }
@media (prefers-color-scheme: dark) { a { color: #00ff9c; } }
"""


def page(lang, title, body):
    return (
        f'<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8">'
        f"<style>{CSS}</style></head><body><h1>{title}</h1>{body}</body></html>\n"
    )


def main():
    for key, brand in BRANDS.items():
        for lang, t in T.items():
            ctx = {"name": brand["name"], "host": brand["host"], "menu": brand["menu"][lang], "docs": DOCS}
            f = {k: v.format(**ctx) for k, v in t.items()}
            out = os.path.join(ROOT, key, "mac", f"{lang}.lproj")
            os.makedirs(out, exist_ok=True)
            welcome = page(lang, f["welcome_title"], f"<p>{f['welcome_body']}</p><p class=\"mute\">{f['close_host']}</p>")
            conclusion = page(
                lang,
                f["done_title"],
                f"<p>{f['how']}</p><ol><li>{f['step1']}</li><li>{f['step2']}</li><li>{f['step3']}</li></ol>"
                f"<p class=\"mute\">{f['help']}</p>",
            )
            with open(os.path.join(out, "welcome.html"), "w") as fh:
                fh.write(welcome)
            with open(os.path.join(out, "conclusion.html"), "w") as fh:
                fh.write(conclusion)


if __name__ == "__main__":
    main()
