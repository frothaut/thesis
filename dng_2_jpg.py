#!/usr/bin/env python3
"""
Einfaches DNG → JPG Batch-Konvertierungs-Skript
mit darktable-cli (für RAW-Entwicklung) und exiftool (für EXIF-Kopie).

Anpassbar über Variablen unten.
"""

import subprocess
from pathlib import Path
import shutil
import sys
from tqdm import tqdm

# =====================================================
# 🧩 CONFIG — hier kannst du alles anpassen
# =====================================================

# Eingabeordner mit .DNG-Dateien
INPUT_DIR = Path("E:/Rothaut_Masterthesis/raw")

# Ausgabeordner für die JPGs
OUTPUT_DIR = Path("E:/Rothaut_Masterthesis/jpg_new")

# Rekursiv durch Unterordner gehen?
RECURSIVE = False

# Vorhandene JPGs überschreiben?
OVERWRITE = True

# JPEG-Qualität (0..100)
JPEG_QUALITY = 100

# ExifTool-Command als frei editierbarer String.
# {src} -> Quell-DNG, {dst} -> Ziel-JPG
EXIFTOOL_CMD = (
    'exiftool -q -q -TagsFromFile "{src}" -all:all -overwrite_original_in_place "{dst}"'
)
# Beispiel für selektive Tags:
# EXIFTOOL_CMD = 'exiftool -q -q -TagsFromFile "{src}" -gps:all -datetimeoriginal -overwrite_original_in_place "{dst}"'

# Tool-Pfade:
# - Wenn None: automatisch via PATH + gängige macOS-Standardorte suchen
# - Oder absolut setzen, z.B.:
#   DARKTABLE_CLI = "/Applications/darktable.app/Contents/MacOS/darktable-cli"
#   EXIFTOOL     = "/opt/homebrew/bin/exiftool"
#DARKTABLE_CLI = "/Applications/darktable.app/Contents/MacOS/darktable-cli" MACOS
DARKTABLE_CLI = "C:/Program Files/darktable/bin/darktable-cli.exe"#windows
#EXIFTOOL     = None
EXIFTOOL = "exiftool.exe" # windows
# =========================


def resolve_tool(name: str, explicit_path: str | None, fallbacks: list[str]) -> str:
    """Ermittelt ausführbaren Pfad: explizit -> PATH -> Fallbacks, sonst Fehler."""
    if explicit_path:
        return explicit_path
    p = shutil.which(name)
    if p:
        return p
    for fb in fallbacks:
        if Path(fb).exists():
            return fb
    raise FileNotFoundError(
        f"Konnte '{name}' nicht finden. Bitte in PATH aufnehmen oder im Script fest setzen."
    )

# <<<<<< Hier dein XMP mit 'embedded metadata' Lens Correction eintragen >>>>>>
# Beispiel: aus der GUI erzeugtes XMP, das nur das Lens-Correction-Modul setzt
#XMP_PATH = Path("/Users/filiprothaut/Documents/HCU/Masterthesis/exif_dummy.DNG.xmp")  # oder None

EXIFTOOL_CMD = (
    'exiftool -q -q -TagsFromFile "{src}" -all:all -overwrite_original_in_place "{dst}"'
)
# --- /CONFIG ---

def convert_with_darktable(src: Path, dst: Path, nr):
    cmd = [
        DARKTABLE_CLI,
        src.as_posix(),
    ]
    n = 3
    if nr <602:
        n = 1
    if nr >601 and nr <828:
        n = 2
    XMP_PATH = Path(f"E:/Rothaut_Masterthesis/thesis/exif_dummy.DNG{n}.xmp")
    if XMP_PATH:
        
        cmd.append(XMP_PATH.as_posix())
    cmd += [
        dst.name,                 # nur der Dateiname!
        "--core",
        "--conf", f"plugins/imageio/format/jpeg/quality={JPEG_QUALITY}",
    ]
    print(f"[darktable] {src.name} -> {dst.name} (embedded lens metadata via XMP)")
    subprocess.run(cmd, check=True, cwd=OUTPUT_DIR)  # hier liegt die Magie
def copy_exif(exiftool: str, src: Path, dst: Path, cmd_template: str) -> None:
    # Ersetzt {src}/{dst} und führt als shell command aus (damit Pipe/Quotes möglich sind)
    # Ersetzt 'exiftool' am Anfang automatisch durch den gefundenen Pfad, falls der String so beginnt.
    cmd_str = cmd_template.format(src=src, dst=dst).strip()
    if cmd_str.startswith("exiftool "):
        # Ersetze führendes "exiftool" durch absoluten Pfad
        cmd_str = cmd_str.replace("exiftool", f'"{exiftool}"', 1)
    print(f"[exiftool] Kopiere Metadaten {src.name} -> {dst.name}")
    subprocess.run(cmd_str, shell=True, check=True)


def main():
    # Tools auf macOS plausibel auflösen
    darktable_cli = resolve_tool(
        "darktable-cli",
        DARKTABLE_CLI,
        fallbacks=[
            "/Applications/darktable.app/Contents/MacOS/darktable-cli",
            "/opt/homebrew/bin/darktable-cli",   # Homebrew (Apple Silicon)
            "/usr/local/bin/darktable-cli",      # Intel/Homebrew alt
        ],
    )
    exiftool = resolve_tool(
        "exiftool",
        EXIFTOOL,
        fallbacks=[
            "/opt/homebrew/bin/exiftool",
            "/usr/local/bin/exiftool",
        ],
    )

    if not INPUT_DIR.exists():
        print(f"❌ INPUT_DIR existiert nicht: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dngs = sorted(INPUT_DIR.rglob("*.DNG") if RECURSIVE else INPUT_DIR.glob("*.DNG"))
    if not dngs:
        print("⚠️  Keine .DNG Dateien gefunden.")
        return

    fails = 0
    for src in tqdm(dngs):
        dst = OUTPUT_DIR / src.with_suffix(".jpg").name
        nr = int(src.name.replace("DJI_", "").replace(".DNG", ""))
        print("NR", nr)
        if dst.exists() and not OVERWRITE:
            print(f"[skip] Existiert: {dst.name}")
            continue
        try:
            convert_with_darktable(src, dst, nr)
            copy_exif(exiftool, src, dst, EXIFTOOL_CMD)
        except subprocess.CalledProcessError as e:
            fails += 1
            print(f"[FAIL] {src.name}: {e}")
        except Exception as e:
            fails += 1
            print(f"[FAIL] {src.name}: {e}")

    ok = len(dngs) - fails
    print(f"\nFertig. Erfolgreich: {ok}/{len(dngs)} | Fehler: {fails}")


if __name__ == "__main__":
    main()