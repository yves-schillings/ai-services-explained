"""Read a local scan with a separately installed Tesseract executable.

Run: python "07_vision_ocr.py" sample.png --language eng
Install the engine and selected language data first. OCR output is unverified text.
"""
from pathlib import Path
import subprocess

def extract(path, language="eng"):
    path=Path(path).resolve(strict=True)
    if path.suffix.lower() not in {".png",".jpg",".jpeg",".tif",".tiff"}:
        raise ValueError("Use a supported local image")
    if not language.replace("+", "").isalpha():raise ValueError("Invalid language code")
    result=subprocess.run(["tesseract",str(path),"stdout","-l",language],
                          check=True,capture_output=True,text=True,timeout=90)
    return result.stdout.strip()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("image");p.add_argument("--language",default="eng");a=p.parse_args()
    print(extract(a.image,a.language))
