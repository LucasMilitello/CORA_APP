"""Base minima para iniciar o CORA usando PySide6."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets

from ..matlab_style_cora import CORAConfig
from ..services import export_service
from ..services.processing_service import run_processing_worker
from .ui.dialogs import select_folder, show_message
from .ui.main_page_qt import MainPageWidget
from .ui.roi_editor import RoiEditorWidget
from .ui.roi_page_qt import RoiPageWidget


_logger = logging.getLogger(__name__)

# PT: DIRECIONAMENTO DO PACOTE qt/core/tk_portable_pack.py | EN: GUIDANCE FOR THE qt/core/tk_portable_pack.py PACKAGE
# PT: Use neste arquivo (orquestrador): | EN: Use the following in this orchestrator file:
# PT: - collect_grouping_entries / build_groups_from_entries: scan + revisao. | EN: - collect_grouping_entries / build_groups_from_entries: scanning and review.
# PT: - all_items: montar fila de processamento em lote. | EN: - all_items: build the batch-processing queue.
# PT: - area_auto_from_processed / closure_vs_0h: metricas por grupo. | EN: - area_auto_from_processed / closure_vs_0h: per-group metrics.
# PT: - OUTPUT_DIR_NAME: fallback da pasta de saida. | EN: - OUTPUT_DIR_NAME: output-folder fallback.
# PT: - TIME_ORDER / GroupingEntry / ProcessedTimepoint: tipos e ordem do dominio. | EN: - TIME_ORDER / GroupingEntry / ProcessedTimepoint: domain types and ordering.
# PT: Deixe regras de negocio aqui e mantenha os arquivos em qt/ui apenas com widgets. | EN: Keep business rules here and restrict qt/ui files to widgets.


class CORAQtApp(QtWidgets.QMainWindow):
    """Janela principal minimalista que orquestra os widgets Qt."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        folder_path: str | None = None,
        output_folder: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CORA (Qt)")
        self.folder_path: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.config = CORAConfig()
        self.processing_worker: Optional[QtCore.QThread] = None
        self._init_ui()
        self._apply_initial_paths(folder_path, output_folder)

    def _init_ui(self) -> None:
        """Configura o layout basico com os widgets em modulos."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter)

        self.main_page = MainPageWidget()
        self.roi_page = RoiPageWidget()
        splitter.addWidget(self.main_page)
        splitter.addWidget(self.roi_page)

        self.roi_editor = RoiEditorWidget(self)

        self.main_page.folder_selected.connect(self._on_folder_selected)
        self.main_page.folder_browse_requested.connect(self._on_browse_folder)
        self.main_page.run_processing_requested.connect(self._on_run_processing)
        self.main_page.export_requested.connect(self._on_export_results)
        self.roi_page.edit_roi_requested.connect(self._on_edit_roi)

    def _on_folder_selected(self, path: str) -> None:
        """Atualiza o estado interno quando o usuario define uma pasta."""
        self.folder_path = Path(path)
        self.main_page.set_folder(str(self.folder_path))
        _logger.info("Pasta de imagens selecionada: %s", path)

    def _on_browse_folder(self) -> None:
        """Mostra dialogo para selecionar pasta e aplica no estado."""
        result = select_folder(self)
        if result:
            self._on_folder_selected(result)

    def _apply_initial_paths(
        self,
        folder_path: str | None,
        output_folder: str | None,
    ) -> None:
        if folder_path:
            self._on_folder_selected(folder_path)
        if output_folder:
            self.output_folder = Path(output_folder)
            if hasattr(self.roi_page, "set_output_folder"):
                self.roi_page.set_output_folder(output_folder)
            _logger.info("Pasta de saida inicial: %s", output_folder)

    def _on_run_processing(self) -> None:
        """Placeholder para chamar o worker de processamento em background."""
        if self.folder_path is None:
            show_message(self, "Informe uma pasta antes de rodar o processamento.")
            return

        # PT: DIRECIONAMENTO AQUI: | EN: GUIDANCE FOR THIS SECTION:
        # 1) discover_image_groups(...)
        # 2) collect_grouping_entries(...)
        # PT: 3) dialog Qt de revisao (QDialog/QTableView) | EN: 3) Qt review dialog (QDialog/QTableView)
        # 4) build_groups_from_entries(...)
        # 5) all_items(...)
        # PT: 6) run_processing_worker + polling com QTimer | EN: 6) run_processing_worker + polling with QTimer
        _ = run_processing_worker  # PT: evita remover import enquanto o fluxo nao e ligado | EN: prevents removing the import before the workflow is connected
        show_message(self, "Fluxo de processamento ainda nao foi implementado no Qt.")

    def _on_export_results(self) -> None:
        """Placeholder para exportacao, mantem hook com os servicos."""

        # PT: DIRECIONAMENTO AQUI: | EN: GUIDANCE FOR THIS SECTION:
        # PT: - OUTPUT_DIR_NAME para saida padrao | EN: - OUTPUT_DIR_NAME for the default output folder
        # PT: - export_service para gravacao de planilha/imagens | EN: - export_service for writing spreadsheets and images
        # PT: - closure_vs_0h / area_auto_from_processed para resumo final | EN: - closure_vs_0h / area_auto_from_processed for the final summary
        _ = export_service  # PT: evita remover import enquanto o fluxo nao e ligado | EN: prevents removing the import before the workflow is connected
        show_message(self, "Exportacao ainda sera ligada ao servico real.")

    def _on_edit_roi(self) -> None:
        """Exibe o editor de ROI quando o botao for acionado."""

        # PT: DIRECIONAMENTO AQUI: | EN: GUIDANCE FOR THIS SECTION:
        # PT: - mover a logica de edicao para qt/ui/roi_editor.py | EN: - move editing logic to qt/ui/roi_editor.py
        # PT: - reutilizar helpers de mascara de qt.core | EN: - reuse mask helpers from qt.core
        self.roi_editor.open()


def qt_main(folder: str | None = None, output: str | None = None) -> None:
    """Inicia a aplicacao Qt."""
    app = QtWidgets.QApplication(sys.argv)
    window = CORAQtApp(folder_path=folder, output_folder=output)
    window.show()
    sys.exit(app.exec())


__all__ = ["CORAQtApp", "qt_main"]
