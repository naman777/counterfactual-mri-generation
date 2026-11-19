"""Single paper location shared by the API and portable demo packager."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_RELATIVE_PATH = Path("research-paper-v11-evaluation/paper_v11_ieee.pdf")
PAPER_PATH = PROJECT_ROOT.parent / PAPER_RELATIVE_PATH
PAPER_DOWNLOAD_NAME = "mrone-paper-v11.pdf"
