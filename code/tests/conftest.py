"""Put ``code/`` on ``sys.path`` so the tests import the library modules directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
