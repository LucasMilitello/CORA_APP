"""Ponto de entrada legado usando Tkinter."""

import sys
from pathlib import Path

# PT: Permite executar este arquivo diretamente (Run File no IDE) sem depender do CWD. | EN: Allows this file to run directly (Run File in the IDE) without depending on the CWD.
_SELF = Path(__file__).resolve()
_PKG_PARENT = str(_SELF.parents[2])  # .../codigo_atual
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

try:
    from cora_projeto.cora_interface import main
except ImportError:
    from cora_interface import main


def run() -> None:
    main()


if __name__ == "__main__":
    run()
