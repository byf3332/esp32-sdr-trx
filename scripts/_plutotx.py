"""Shared by the scripts that use pluto-tx (GPL-3) as it is installed on this machine: find it, give the process the library path that gr-m17
needs (it is read when the process starts, so the script starts itself again once), and put pluto-tx on sys.path."""
import os
import sys
from pathlib import Path

DEFAULT_DIR = Path.home() / "Dokumente/plutosdr"
LIBS = [str(Path.home() / ".local/lib/x86_64-linux-gnu")]


def prepare(pluto_tx_dir=DEFAULT_DIR):
    if os.environ.get("ESPDR_PLUTOTX_ENV") != "1":
        libs = LIBS + [str(Path(pluto_tx_dir) / "rade_c/build/src")]
        env = dict(os.environ, ESPDR_PLUTOTX_ENV="1", LD_LIBRARY_PATH=":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")]))
        os.execve(sys.executable, [sys.executable] + sys.argv, env)
    sys.path.insert(0, str(pluto_tx_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
