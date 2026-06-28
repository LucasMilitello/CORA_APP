"""Componente Matplotlib que substitui a janela Tk de edicao de ROI."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets

# PT: DIRECIONAMENTO: | EN: GUIDANCE:
# PT: - Este arquivo recebe a logica que hoje esta em _open_roi_editor no Tk. | EN: - This file receives the logic currently implemented in Tk's _open_roi_editor.
# PT: - Helpers que devem ser usados aqui (vindos de qt.core): | EN: - Helpers to use here (from qt.core):
#   mask_to_polygon, polygon_to_mask, resize_mask, offset_mask_sdf, keep_largest_component.
# PT: - O orquestrador (cora_interface_qt.py) so abre este widget/dialog. | EN: - The orchestrator (cora_interface_qt.py) only opens this widget/dialog.


class RoiEditorWidget(QtWidgets.QWidget):
    """Canvas dentro do Qt que encapsula a edicao de mascara."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def open(self) -> None:
        """Abre o widget (pode virar QDialog depois)."""
        self.show()

    def load_image(self, path: Path) -> None:
        """Placeholder para desenhar imagem e ligar eventos Matplotlib."""
        axes = self.canvas.figure.subplots()
        axes.clear()
        axes.text(0.5, 0.5, f"Carregar imagem: {path.name}", ha="center", va="center")
        self.canvas.draw()
