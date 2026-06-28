"""Helper para iniciar a interface Qt (pluggable por scripts)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_qt_main():
    """Resolve o entrypoint tanto via modulo quanto via execucao direta."""
    if __package__:
        from .cora_interface_qt import qt_main as resolved_qt_main

        return resolved_qt_main

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    from cora_projeto.qt.cora_interface_qt import qt_main as resolved_qt_main

    return resolved_qt_main


qt_main = _resolve_qt_main()


def main() -> None:
    """Executa o loop do Qt."""
    parser = argparse.ArgumentParser(description="Inicia CORA com PySide6.")
    parser.add_argument("--folder", type=str, help="Pasta inicial de imagens.")
    parser.add_argument("--output", type=str, help="Pasta de saida opcional.")
    args = parser.parse_args()
    qt_main(folder=args.folder, output=args.output)


if __name__ == "__main__":
    main()
