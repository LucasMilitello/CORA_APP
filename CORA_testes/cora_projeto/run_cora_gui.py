"""Ponto de entrada da interface grafica da aplicacao CORA."""

import importlib.util
import os
from pathlib import Path
import sys


REQUIRED_IMPORTS = ("cv2", "matplotlib", "numpy", "psutil", "scipy", "sklearn", "tifffile")


def _resolve_venv_python(project_root: Path) -> Path | None:
    """Localiza o Python da .venv local ou da pasta pai quando existir."""
    for base_dir in (project_root, project_root.parent):
        candidates = (
            base_dir / ".venv" / "Scripts" / "python.exe",
            base_dir / ".venv" / "bin" / "python",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _missing_required_imports() -> list[str]:
    """Retorna dependencias externas ausentes no interpretador atual."""
    return [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]


def _bootstrap_project_python() -> None:
    """Permite executar este arquivo diretamente mesmo com outro Python selecionado."""
    if getattr(sys, "frozen", False):
        return

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    missing_imports = _missing_required_imports()
    if not missing_imports:
        return

    venv_python = _resolve_venv_python(project_root)
    current_python = Path(sys.executable).resolve()
    if venv_python is not None and current_python != venv_python.resolve():
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])

    missing_text = ", ".join(missing_imports)
    requirements_path = project_root / "requirements.txt"
    python_hint = venv_python or project_root / ".venv" / "Scripts" / "python.exe"
    raise ModuleNotFoundError(
        "Dependencias ausentes no Python atual: "
        f"{missing_text}. Execute com {python_hint} "
        f"ou instale as dependencias com: {sys.executable} -m pip install -r {requirements_path}"
    )


_bootstrap_project_python()


def _resolve_main():
    """Resolve o entrypoint tanto via modulo quanto via execucao direta."""
    if __package__:
        from .cora_interface import main as resolved_main
        return resolved_main

    package_dir = Path(__file__).resolve().parent
    package_parent = package_dir.parent
    package_parent_str = str(package_parent)
    if package_parent_str not in sys.path:
        sys.path.insert(0, package_parent_str)

    from cora_projeto.cora_interface import main as resolved_main
    return resolved_main


main = _resolve_main()


if __name__ == "__main__":
    main()
