"""Assemble captured dashboard views, not live GPU inference. Requires ffmpeg."""
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
scenes = [
    ("01-overview", "MR-one: saved-evidence walkthrough\nRecorded dashboard views. No live GPU inference is shown."),
    ("02-findings", "The research question and pipeline\nHeadline rates name each judge and include patient-clustered confidence intervals."),
    ("03-studio", "Studio: choose a slice and an acquisition target\nThe paper seed is selected automatically. EchoTime stays at its source value."),
    ("04-explanations", "Explainability: difference and local SSIM maps\nInspect where the target edit changes the image."),
    ("05-histogram", "Intensity distribution on source foreground pixels\nThe same foreground mask applies to the source and edited image."),
    ("06-verification", "Independent verification and the same-noise null edit\nJudge disagreement remains visible. Agreement does not establish clinical validity."),
    ("07-results", "Results: cohort rates, uncertainty and provenance\nRead the original predictor and held-out judges separately."),
    ("08-mechanism", "Mechanism: high-frequency energy and uncertainty\nReadable labels and two-decimal values retain the original units."),
    ("09-evidence", "Evidence Explorer: browse saved evaluation outputs\nFilter successes and disagreements. Selected demo cases are illustrative."),
    ("10-export", "Saved-patient export: images and a provenance manifest\nWhole-patient live generation remains pending. Research use only."),
]
def stamp(seconds):
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d},000"
concat, captions = [], []
for i, (name, caption) in enumerate(scenes):
    assert (root / f"frames/{name}.jpg").is_file()
    concat += [f"file 'frames/{name}.jpg'", "duration 18"]
    captions.append(f"{i+1}\n{stamp(i*18)} --> {stamp((i+1)*18)}\n{caption}\n")
concat.append(f"file 'frames/{scenes[-1][0]}.jpg'")
(root / "frames.concat").write_text("\n".join(concat), encoding="utf-8")
(root / "captions.srt").write_text("\n".join(captions), encoding="utf-8")
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    raise RuntimeError("ffmpeg is required")
subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "warning", "-n", "-f", "concat", "-safe", "0", "-i", "frames.concat",
    "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,subtitles=captions.srt:force_style='FontName=Arial,FontSize=18,BorderStyle=3,Outline=1,Shadow=0,MarginV=12'",
    "-t", "180", "-r", "24", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "mrone-backup-walkthrough.mp4"], cwd=root, check=True)
