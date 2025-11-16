from pathlib import Path
import subprocess

# Adjust these to your setup
DARKTABLE_CLI = "C:/Program Files/darktable/bin/darktable-cli.exe"#windows
INPUT_DIR = Path(r"E:\Rothaut_Masterthesis\thesis\raw_masks/bearbeitet")
OUTPUT_DIR = Path(r"E:\Rothaut_Masterthesis\masks_new")
XMP_PATH = Path(r"E:/Rothaut_Masterthesis/thesis/exif_dummy.DNG3.xmp")  # XMP with only distortion correction enabled


def convert_with_darktable(src: Path, dst: Path, xmp_path: Path):
    """
    Apply only the settings contained in xmp_path (e.g. just distortion correction)
    to src and export to dst using darktable-cli.
    """
    cmd = [
        DARKTABLE_CLI,
        src.as_posix(),
    ]

    # only add XMP if it actually exists
    if xmp_path.is_file():
        cmd.append(xmp_path.as_posix())

    # output filename ONLY (darktable-cli will write into cwd)
    cmd.append(dst.name)

    print(f"[darktable] {src.name} -> {dst.name} (XMP: {xmp_path.name})")
    subprocess.run(cmd, check=True, cwd=dst.parent)


def process_png_folder(input_dir: Path, output_dir: Path, xmp_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))

    if not png_files:
        print(f"No PNGs found in {input_dir}")
        return

    for src in png_files:
        # keep same filename and extension in the output folder
        dst = output_dir / (src.name)
        convert_with_darktable(src, dst, xmp_path)


if __name__ == "__main__":
    process_png_folder(INPUT_DIR, OUTPUT_DIR, XMP_PATH)