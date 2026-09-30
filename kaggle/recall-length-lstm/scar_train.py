import sys
import subprocess
from pathlib import Path

REPO_DIR = Path("/kaggle/working/scar")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "research/v3", "--single-branch", "https://github.com/arjunkshah12345-hash/scar.git", str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR / "kaggle" / "recall-length-split"))
from scar_train_common import run_model

run_model("lstm")
