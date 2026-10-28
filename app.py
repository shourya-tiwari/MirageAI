"""MirageAI dashboard launcher.

    streamlit run app.py

Runs the dashboard in app/streamlit_app.py, so it can be started from the
repository root without remembering the internal path.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "streamlit_app.py"), run_name="__main__")
