"""Check publication source preservation and numerical contracts."""
from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Run release checks, including in an extracted standalone archive."""
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                   cwd=root, check=True)


if __name__ == "__main__":
    main()
