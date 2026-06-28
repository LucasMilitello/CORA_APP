"""Funções utilitárias para substituir os diálogos do Tkinter."""

from PySide6 import QtWidgets


def select_folder(parent: QtWidgets.QWidget | None = None) -> str | None:
    """Abre um diálogo para escolher pasta."""
    folder = QtWidgets.QFileDialog.getExistingDirectory(parent, "Escolha a pasta de imagens")
    return folder or None


def show_message(parent: QtWidgets.QWidget | None, message: str, title: str = "CORA") -> None:
    """Mostra uma mensagem de informação simples."""
    QtWidgets.QMessageBox.information(parent, title, message)
