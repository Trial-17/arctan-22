"""Génère les visuels des installeurs (Windows Inno Setup + macOS productbuild).

Usage : python3 installer_assets/make_assets.py
Sorties : installer_assets/<marque>/ (BMP assistant, ICO, fond Mac en TIFF Retina).
Les fichiers générés sont commités : le CI ne relance pas ce script.
"""
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
LOGO = os.path.join(ROOT, "logo.png")
# Geist (variable), la police du site premierecopilot.com
GEIST = os.path.join(ROOT, "fonts/Geist.woff2")

# Fond uni du thème sombre du site (--color-bg), sans dégradé
BG = (10, 10, 10)
FG_SOFT = (163, 163, 163)  # --color-fg-soft
GREEN = (0, 255, 156)

BRANDS = {
    "premiere": {"name": "PremiereCopilot", "tagline": ["AI Copilot for", "Adobe Premiere Pro"]},
    "davinci": {"name": "DavinciClaude", "tagline": ["Claude, native inside", "DaVinci Resolve"]},
}


def font(weight, size):
    f = ImageFont.truetype(GEIST, size)
    f.set_variation_by_name(weight)
    return f


def paste_logo(im, size, cx, cy):
    logo = Image.open(LOGO).convert("RGBA").resize((size, size), Image.LANCZOS)
    im.paste(logo, (int(cx - size / 2), int(cy - size / 2)), logo)


def wizard_image(brand, scale):
    """Panneau gauche de l'assistant Windows (164x314 à 100 %)."""
    w, h = round(164 * scale), round(314 * scale)
    im = Image.new("RGB", (w, h), BG)
    paste_logo(im, round(76 * scale), w / 2, h * 0.36)
    d = ImageDraw.Draw(im)
    f_name = font(b"SemiBold", round(15.5 * scale))
    f_tag = font(b"Regular", round(10.5 * scale))
    y = h * 0.58
    d.text((w / 2, y), brand["name"], font=f_name, fill=(250, 250, 250), anchor="mm")
    y += 22 * scale
    for line in brand["tagline"]:
        d.text((w / 2, y), line, font=f_tag, fill=FG_SOFT, anchor="mm")
        y += 14 * scale
    d.rectangle((w * 0.38, h - 18 * scale, w * 0.62, h - 16.5 * scale), fill=GREEN)
    return im


def small_image(scale):
    """Vignette en haut à droite des pages intérieures (55x55 à 100 %)."""
    s = round(55 * scale)
    im = Image.new("RGB", (s, s), (255, 255, 255))
    paste_logo(im, s, s / 2, s / 2)
    return im


def mac_background(scale):
    """Fond de l'installeur macOS : logo en bas à gauche, sous la liste des étapes."""
    w, h = round(180 * scale), round(150 * scale)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    logo = Image.open(LOGO).convert("RGBA").resize((round(72 * scale),) * 2, Image.LANCZOS)
    im.paste(logo, (round(26 * scale), h - round(98 * scale)), logo)
    return im


def main():
    for key, brand in BRANDS.items():
        out = os.path.join(ROOT, key)
        os.makedirs(out, exist_ok=True)
        for scale, suffix in ((1, "100"), (1.5, "150"), (2, "200")):
            wizard_image(brand, scale).save(os.path.join(out, f"wizard_{suffix}.bmp"))
            small_image(scale).save(os.path.join(out, f"wizard_small_{suffix}.bmp"))
        Image.open(LOGO).convert("RGBA").save(
            os.path.join(out, "setup.ico"),
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
        one, two = os.path.join(out, "bg.png"), os.path.join(out, "bg@2x.png")
        mac_background(1).save(one)
        mac_background(2).save(two)
        subprocess.run(
            ["tiffutil", "-cathidpicheck", one, two, "-out", os.path.join(out, "background.tiff")],
            check=True, capture_output=True,
        )
        os.remove(one)
        os.remove(two)
        wizard_image(brand, 2).save(os.path.join(out, "preview_wizard.png"))


if __name__ == "__main__":
    main()
