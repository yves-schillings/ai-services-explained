"""Transcribe a local recording using separately provisioned Whisper weights.

Run: python "08_speech.py" recording.wav --model-dir PATH --language en
Requires faster-whisper. Model loading is local only. No automatic download.
"""
from pathlib import Path

def transcribe(audio, model_dir, language="en"):
    audio=Path(audio).resolve(strict=True);model_dir=Path(model_dir).resolve(strict=True)
    if not model_dir.is_dir():raise ValueError("Expected a local model directory")
    from faster_whisper import WhisperModel
    model=WhisperModel(str(model_dir),device="cpu",compute_type="int8",local_files_only=True)
    segments, info=model.transcribe(str(audio),language=language,beam_size=1)
    return [{"start":s.start,"end":s.end,"text":s.text} for s in segments]

if __name__=="__main__":
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument("audio");p.add_argument("--model-dir",required=True);p.add_argument("--language",default="en");a=p.parse_args()
    print(json.dumps({"draft_segments":transcribe(a.audio,a.model_dir,a.language),"human_review_required":True},indent=2))
