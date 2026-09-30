import subprocess
import sys
from pathlib import Path

repo = Path("/kaggle/working/scar")
if not repo.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "research/v3", "--single-branch", "https://github.com/arjunkshah12345-hash/scar.git", str(repo)], check=True)
sys.path.insert(0, str(repo / "kaggle" / "recall-length-split"))
from scar_train_common import run_model

run_model("transformer", [0])
