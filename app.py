import runpy
import sys
from pathlib import Path

# Ensure workspace root is in path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Launch frontend app
runpy.run_path(str(ROOT / "frontend" / "app.py"), run_name="__main__")
