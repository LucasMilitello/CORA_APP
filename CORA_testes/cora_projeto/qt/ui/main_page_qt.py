"""Configuracao principal da interface Qt."""

from PySide6 import QtCore, QtWidgets

# PT: DIRECIONAMENTO: | EN: GUIDANCE:
# PT: - Este arquivo deve ficar apenas com UI (campos, botoes, sinais). | EN: - This file should contain UI only (fields, buttons, and signals).
# PT: - Nao colocar aqui logica de agrupamento/processamento/exportacao. | EN: - Do not place grouping, processing, or export logic here.
# PT: - Regras de negocio vao para: | EN: - Business rules belong in:
# PT:   1) qt/cora_interface_qt.py (orquestracao) | EN:   1) qt/cora_interface_qt.py (orchestration)
# PT:   2) qt/core/tk_portable_pack.py (funcoes puras reaproveitadas do Tk) | EN:   2) qt/core/tk_portable_pack.py (pure functions reused from Tk)


class MainPageWidget(QtWidgets.QWidget):
    """Widget com campos de pasta, botoes e sinais para o app principal."""

    folder_selected = QtCore.Signal(str)
    folder_browse_requested = QtCore.Signal()
    run_processing_requested = QtCore.Signal()
    export_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(10)

        layout.addWidget(QtWidgets.QLabel("Pasta de imagens:"), 0, 0)
        self.folder_input = QtWidgets.QLineEdit()
        self.folder_input.setPlaceholderText("Selecione ou digite o caminho da pasta")
        layout.addWidget(self.folder_input, 0, 1)
        browse_btn = QtWidgets.QPushButton("Procurar pasta")
        browse_btn.clicked.connect(self.folder_browse_requested.emit)
        layout.addWidget(browse_btn, 0, 2)

        layout.addWidget(QtWidgets.QLabel("Pasta de saida (opcional):"), 1, 0)
        self.output_input = QtWidgets.QLineEdit()
        self.output_input.setPlaceholderText("Use outro local para salvar resultados")
        layout.addWidget(self.output_input, 1, 1, 1, 2)

        self.run_btn = QtWidgets.QPushButton("Executar processamento")
        self.run_btn.clicked.connect(self.run_processing_requested.emit)
        self.export_btn = QtWidgets.QPushButton("Exportar resultados")
        self.export_btn.clicked.connect(self.export_requested.emit)

        layout.addWidget(self.run_btn, 2, 0, 1, 3)
        layout.addWidget(self.export_btn, 3, 0, 1, 3)

        self.folder_input.editingFinished.connect(
            lambda: self.folder_selected.emit(self.folder_input.text().strip())
        )

    def set_folder(self, path: str) -> None:
        """Preenche o campo de pasta sem disparar sinais."""
        self.folder_input.blockSignals(True)
        self.folder_input.setText(path)
        self.folder_input.blockSignals(False)
