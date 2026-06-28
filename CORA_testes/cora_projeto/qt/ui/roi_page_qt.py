"""Controles adicionais de visualizacao e edicao de mascara."""

from PySide6 import QtCore, QtWidgets

# PT: DIRECIONAMENTO: | EN: GUIDANCE:
# PT: - Este arquivo eh so a "face" de controle da tela de ROI. | EN: - This file is only the control-facing layer of the ROI screen.
# PT: - O mapeamento real de preview (contour/filled/per_time) deve ser feito no orquestrador qt/cora_interface_qt.py usando presets de qt.core.
# EN: - Actual preview mapping (contour/filled/per_time) should be implemented in the qt/cora_interface_qt.py orchestrator using qt.core presets.


class RoiPageWidget(QtWidgets.QWidget):
    """Widget secundario com botoes para editar e alternar preview."""

    edit_roi_requested = QtCore.Signal()
    toggle_preview_requested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(QtWidgets.QLabel("Editor de ROI"))
        edit_btn = QtWidgets.QPushButton("Abrir editor de ROI")
        edit_btn.clicked.connect(self.edit_roi_requested.emit)
        layout.addWidget(edit_btn)

        self.preview_combo = QtWidgets.QComboBox()
        self.preview_combo.addItems(["Contorno", "Area preenchida"])
        self.preview_combo.currentTextChanged.connect(self.toggle_preview_requested.emit)
        layout.addWidget(self.preview_combo)
