import subprocess
import sys
from pathlib import Path


PATTERNS_DIR = Path(__file__).parent
GENERATOR = PATTERNS_DIR / "gen_pattern.py"



OUTPUT_DIR = PATTERNS_DIR.parent / "generated"
OUTPUT_DIR.mkdir(exist_ok=True)


subprocess.run(
    [
        sys.executable,
        str(GENERATOR),
        "-o", str(OUTPUT_DIR / "chess.svg"),
        "--rows", "9",
        "--columns", "6",
        "--type", "checkerboard",
        "--square_size", "20",
        "--page_size", "A4",
    ],
    check=True,
)