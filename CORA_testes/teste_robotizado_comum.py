"""Infraestrutura compartilhada pelos testes robotizados do CORA.

O processo fica consciente do DPI configurado no Windows, sem alterar escala, DPI ou
velocidade do ponteiro. Cada arquivo de teste define seu proprio fluxo de interacao.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import types
from typing import Callable
import unicodedata


ACTION_DELAY_SECONDS = 1.0
SAVE_SELECTION_DELAY_SECONDS = ACTION_DELAY_SECONDS
EDIT_IMAGE_INTERVAL_SECONDS = 15.0
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
ROBOT_TEST_FOLDER = Path(r"C:\Users\milit\Desktop\Teste_App\Robot_test")
MAIN_WINDOW_RE = r".*(CORA|CORA).*"

ROBOTIZED_BUTTONS = (
    "Robot test",
    "Abrir aplicativo para uso normal",
    "Open application for normal use",
)
LOAD_BUTTONS = ("Carregar grupos", "Load groups")
MARK_ALL_BUTTONS = ("Marcar todos", "Select all")
AUTO_FILL_BUTTONS = ("Auto preencher tempos vazios", "Auto fill missing times")
START_BUTTONS = ("Iniciar processamento", "Start processing")
NEXT_GROUP_BUTTONS = ("Proximo grupo", "Next group")
EDIT_MASKS_BUTTONS = ("Editar mascaras", "Edit masks")
SAVE_BUTTONS = ("Salvar resultado", "Save result")
YES_BUTTONS = ("Sim", "Yes")
OK_BUTTONS = ("OK",)


@dataclass
class ActionRecord:
    timestamp: str
    action: str


@dataclass
class RobotizedTestContext:
    application: object
    main_window: object
    robot: "HumanRobot"
    source_images: list[Path]
    processing_timeout: float


def _normalize(text: object) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(value.replace("&", "").lower().split())


def configure_windows_dpi() -> tuple[int | None, int | None]:
    """Usa o DPI do sistema e apenas consulta a velocidade atual do mouse."""
    if os.name != "nt":
        raise RuntimeError("Este teste robotizado requer Windows.")

    try:
        # PT: PROCESS_SYSTEM_DPI_AWARE: coordenadas seguem a escala configurada no Windows. | EN: PROCESS_SYSTEM_DPI_AWARE: coordinates follow the scale configured in Windows.
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    dpi: int | None = None
    try:
        dpi = int(ctypes.windll.user32.GetDpiForSystem())
    except Exception:
        pass

    mouse_speed = ctypes.c_int()
    try:
        # PT: SPI_GETMOUSESPEED somente consulta; nenhuma configuracao do usuario e alterada. | EN: SPI_GETMOUSESPEED only queries the value; no user setting is changed.
        ok = ctypes.windll.user32.SystemParametersInfoW(0x0070, 0, ctypes.byref(mouse_speed), 0)
        speed_value = int(mouse_speed.value) if ok else None
    except Exception:
        speed_value = None
    return dpi, speed_value


def ensure_pywinauto_interpreter() -> None:
    """Reabre o teste no .venv automaticamente quando o VS Code usa outro Python."""
    if importlib.util.find_spec("pywinauto") is not None:
        return
    tests_root = Path(__file__).resolve().parent
    requirements_path = tests_root / "requirements_robotizado.txt"
    venv_python = tests_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.is_file() and Path(sys.executable).resolve() != venv_python.resolve():
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise RuntimeError(
        "pywinauto nao esta instalado no Python atual. Execute: "
        f"{sys.executable} -m pip install -r {requirements_path}"
    )


def configure_comtypes_cache() -> None:
    """Forca o cache do comtypes para uma pasta temporaria gravavel."""
    cache_dir = Path(tempfile.gettempdir()) / "CORA" / "comtypes_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import comtypes

    gen_module = sys.modules.get("comtypes.gen")
    if gen_module is None:
        gen_module = types.ModuleType("comtypes.gen")
        sys.modules["comtypes.gen"] = gen_module
    gen_module.__path__ = [str(cache_dir)]
    comtypes.gen = gen_module


class HumanRobot:
    """Centraliza a pausa obrigatoria e o registro de cada acao humana."""

    def __init__(self, keyboard_module) -> None:
        self.keyboard = keyboard_module
        self.records: list[ActionRecord] = []

    def _finish_action(self, description: str, delay_after: float = ACTION_DELAY_SECONDS) -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.records.append(ActionRecord(timestamp=stamp, action=description))
        print(f"[{stamp}] {description}", flush=True)
        time.sleep(max(0.0, float(delay_after)))

    def click(
        self,
        control,
        description: str,
        delay_after: float = ACTION_DELAY_SECONDS,
    ) -> None:
        control.click_input()
        self._finish_action(description, delay_after=delay_after)

    def click_at(
        self,
        control,
        coordinates: tuple[int, int],
        description: str,
        delay_after: float = ACTION_DELAY_SECONDS,
    ) -> None:
        control.click_input(coords=coordinates)
        self._finish_action(description, delay_after=delay_after)

    def keys(
        self,
        keys: str,
        description: str,
        delay_after: float = ACTION_DELAY_SECONDS,
    ) -> None:
        self.keyboard.send_keys(keys, pause=0.0, with_spaces=True)
        self._finish_action(description, delay_after=delay_after)

    def maximize(self, window) -> None:
        window.maximize()
        self._finish_action("Maximizar a janela principal")

    def wait_for_image_edit(self, image_number: int) -> None:
        """Simula o tempo humano de edicao antes de confirmar cada imagem."""
        self._finish_action(
            f"Aguardar {EDIT_IMAGE_INTERVAL_SECONDS:.1f}s para editar a imagem {image_number}",
            delay_after=EDIT_IMAGE_INTERVAL_SECONDS,
        )


def _visible_descendants(scope) -> list:
    controls = []
    try:
        descendants = scope.descendants()
    except Exception:
        return controls
    for control in descendants:
        try:
            if control.is_visible():
                controls.append(control)
        except Exception:
            continue
    return controls


def find_control(
    scope,
    titles: tuple[str, ...],
    timeout: float = 20.0,
    require_enabled: bool | None = None,
):
    expected = {_normalize(title) for title in titles}
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        for control in _visible_descendants(scope):
            try:
                if _normalize(control.window_text()) not in expected:
                    continue
                if require_enabled is not None and bool(control.is_enabled()) != require_enabled:
                    continue
                return control
            except Exception:
                continue
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Controle nao encontrado: {titles}")
        time.sleep(0.2)


def try_find_control(scope, titles: tuple[str, ...], require_enabled: bool | None = None):
    try:
        return find_control(scope, titles, timeout=0.0, require_enabled=require_enabled)
    except TimeoutError:
        return None


def _find_robot_test_button_coordinates(main_window) -> tuple[int, int] | None:
    """Localiza visualmente o botao inferior quando Tk nao o expoe ao pywinauto."""
    try:
        import cv2
        import numpy as np

        screenshot = np.asarray(main_window.capture_as_image().convert("RGB"), dtype=np.uint8)
    except Exception:
        return None

    # PT: Cores de botoes dos temas disponibilizados pelo CORA. | EN: Button colors for the themes provided by CORA.
    button_colors = (
        (232, 238, 247),  # standard_light
        (49, 59, 74),  # standard_dark
        (138, 33, 78),  # rose_light
        (232, 165, 176),  # rose_dark
        (31, 93, 133),  # blue_light
        (110, 168, 215),  # blue_dark
        (47, 107, 79),  # green_light
        (139, 191, 159),  # green_dark
        (106, 61, 143),  # purple_light
        (183, 148, 214),  # purple_dark
        (68, 84, 106),  # graphite_light
        (174, 183, 194),  # graphite_dark
    )
    image_height, image_width = screenshot.shape[:2]
    candidates: list[tuple[int, int, int, int, int]] = []
    for color in button_colors:
        target = np.asarray(color, dtype=np.int16)
        difference = np.max(np.abs(screenshot.astype(np.int16) - target), axis=2)
        mask = (difference <= 6).astype(np.uint8)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for index in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[index])
            rectangle_area = max(width * height, 1)
            if width < 60 or height < 20:
                continue
            if width > int(image_width * 0.75) or height > int(image_height * 0.15):
                continue
            if area < 500 or (float(area) / float(rectangle_area)) < 0.50:
                continue
            candidates.append((x, y, width, height, area))

    if not candidates:
        return None

    # PT: Na tela de modos, Robot test e o ultimo botao verticalmente. | EN: On the modes screen, Robot test is the last button vertically.
    x, y, width, height, _area = max(
        candidates,
        key=lambda item: (item[1] + item[3], item[0] + item[2]),
    )
    return (x + width // 2, y + height // 2)


def open_robot_test_page(main_window, robot: HumanRobot) -> None:
    """Abre Robot test por acessibilidade, imagem ou atalho, nessa ordem."""
    try:
        button = find_control(
            main_window,
            ROBOTIZED_BUTTONS,
            timeout=2.0,
            require_enabled=True,
        )
    except TimeoutError:
        button = None

    if button is not None:
        robot.click(button, "Abrir Robot test pelo nome do botao")
        return

    coordinates = _find_robot_test_button_coordinates(main_window)
    if coordinates is not None:
        print(f"Botao Robot test localizado visualmente em {coordinates}.", flush=True)
        robot.click_at(
            main_window,
            coordinates,
            "Abrir Robot test pela posicao identificada na imagem",
        )
        return

    main_window.set_focus()
    robot.keys("{F2}", "Abrir Robot test pelo atalho F2")


def find_visible_edits(main_window, count: int = 2, timeout: float = 20.0) -> list:
    deadline = time.monotonic() + timeout
    while True:
        edits = []
        for control in _visible_descendants(main_window):
            try:
                friendly = _normalize(control.friendly_class_name())
                class_name = _normalize(control.class_name())
                if friendly == "edit" or class_name == "edit":
                    if control.is_enabled():
                        edits.append(control)
            except Exception:
                continue
        edits.sort(key=lambda item: (item.rectangle().top, item.rectangle().left))
        if len(edits) >= count:
            return edits[:count]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Esperados {count} campos de texto; encontrados {len(edits)}.")
        time.sleep(0.2)


def set_clipboard_text(text: str) -> None:
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, str(text))
    finally:
        win32clipboard.CloseClipboard()


def paste_into_edit(robot: HumanRobot, edit, text: str, label: str) -> None:
    robot.click(edit, f"Clicar no campo {label}")
    robot.keys("^a", f"Selecionar o conteudo do campo {label}")
    set_clipboard_text(text)
    robot.keys("^v", f"Colar {label}")


def set_robot_field_from_clipboard(
    main_window,
    robot: HumanRobot,
    shortcut: str,
    text: str,
    label: str,
    confirmation_marker: str,
) -> None:
    """Envia um caminho ao Tk e somente retorna apos a confirmacao do aplicativo."""
    set_clipboard_text(text)
    for attempt in range(1, 4):
        main_window.set_focus()
        robot.keys(shortcut, f"Preencher {label} (tentativa {attempt})")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                title = str(main_window.window_text())
            except Exception:
                title = ""
            if confirmation_marker in title:
                print(f"Campo {label} confirmado pelo CORA.", flush=True)
                return
            time.sleep(0.2)
    raise RuntimeError(f"O CORA nao confirmou o preenchimento do campo {label}.")


def visible_dialogs(application) -> list:
    dialogs = []
    try:
        windows = application.windows()
    except Exception:
        return dialogs
    for window in windows:
        try:
            if window.is_visible() and window.class_name() == "#32770":
                dialogs.append(window)
        except Exception:
            continue
    return dialogs


def dialog_description(dialog) -> str:
    parts = []
    try:
        parts.append(dialog.window_text())
    except Exception:
        pass
    for child in _visible_descendants(dialog):
        try:
            text = child.window_text().strip()
            if text and text not in parts:
                parts.append(text)
        except Exception:
            pass
    return " | ".join(parts)


def answer_dialog(robot: HumanRobot, dialog, prefer_yes: bool = False) -> str:
    description = dialog_description(dialog)
    button_titles = YES_BUTTONS if prefer_yes else OK_BUTTONS
    button = try_find_control(dialog, button_titles, require_enabled=True)
    if button is None and prefer_yes:
        button = try_find_control(dialog, OK_BUTTONS, require_enabled=True)
    if button is not None:
        robot.click(button, f"Responder dialogo: {description}")
    else:
        dialog.set_focus()
        robot.keys("{ENTER}", f"Confirmar dialogo: {description}")
    return description


def wait_and_answer_dialog(
    application,
    robot: HumanRobot,
    prefer_yes: bool,
    timeout: float = 30.0,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        dialogs = visible_dialogs(application)
        if dialogs:
            return answer_dialog(robot, dialogs[0], prefer_yes=prefer_yes)
        time.sleep(0.2)
    raise TimeoutError("O dialogo esperado nao apareceu.")


def dismiss_information_dialogs(application, robot: HumanRobot) -> list[str]:
    dismissed = []
    while True:
        dialogs = visible_dialogs(application)
        if not dialogs:
            return dismissed
        dismissed.append(answer_dialog(robot, dialogs[0], prefer_yes=False))


def progress_window_visible(application) -> bool:
    try:
        windows = application.windows()
    except Exception:
        return False
    keywords = ("processamento", "processing", "aplicando", "applying")
    for window in windows:
        try:
            title = _normalize(window.window_text())
            if window.is_visible() and any(keyword in title for keyword in keywords):
                return True
        except Exception:
            continue
    return False


def wait_for_processing_cycle(
    application,
    main_window,
    robot: HumanRobot,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    saw_busy = False
    started_at = time.monotonic()
    while time.monotonic() < deadline:
        dismiss_information_dialogs(application, robot)
        save_button = try_find_control(main_window, SAVE_BUTTONS)
        processing_visible = progress_window_visible(application)
        if processing_visible:
            saw_busy = True
        elif saw_busy:
            return
        if save_button is not None:
            try:
                if not save_button.is_enabled():
                    saw_busy = True
                elif saw_busy or (time.monotonic() - started_at) >= 5.0:
                    return
            except Exception:
                pass
        time.sleep(0.25)
    raise TimeoutError("O processamento nao terminou dentro do tempo configurado.")


def visit_all_groups(main_window, robot: HumanRobot, maximum: int) -> int:
    """Percorre grupos pelo atalho Tk e valida cada avanco pelo titulo da janela."""
    marker_re = re.compile(r"\[ROBOT_GROUP:(\d+)/(\d+)\]")
    previous_position = 0
    for attempt in range(1, int(maximum) + 1):
        main_window.set_focus()
        robot.keys("+{F6}", f"Abrir o proximo grupo (acao {attempt})")
        deadline = time.monotonic() + 3.0
        position = 0
        total = 0
        while time.monotonic() < deadline:
            try:
                title = str(main_window.window_text())
            except Exception:
                title = ""
            match = marker_re.search(title)
            if match is not None:
                position, total = (int(value) for value in match.groups())
                if position != previous_position or position >= total:
                    break
            time.sleep(0.2)
        else:
            raise RuntimeError("O CORA nao confirmou o avanco entre grupos.")

        if total < 1 or position < 1 or position > total:
            raise RuntimeError(f"Posicao de grupo invalida: {position}/{total}.")
        if total > int(maximum):
            raise RuntimeError(f"Quantidade de grupos acima do limite: {total}/{maximum}.")
        if position >= total:
            return total
        previous_position = position

    raise RuntimeError("A navegacao excedeu a quantidade maxima esperada de grupos.")


def _checkbox_is_checked(control) -> bool:
    for method_name in ("get_check_state", "is_checked"):
        method = getattr(control, method_name, None)
        if not callable(method):
            continue
        try:
            return bool(method())
        except Exception:
            continue
    raise RuntimeError(f"Nao foi possivel consultar o checkbox: {control.window_text()}")


def find_save_checkboxes(main_window, timeout: float = 20.0) -> list:
    """Localiza somente a primeira linha 0h/24h/48h: 'Incluir no salvamento'."""
    expected_titles = {"0h", "24h", "48h"}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidates = []
        for control in _visible_descendants(main_window):
            try:
                if _normalize(control.window_text()) not in expected_titles:
                    continue
                friendly = _normalize(control.friendly_class_name())
                if "check" not in friendly:
                    continue
                if control.is_enabled():
                    candidates.append(control)
            except Exception:
                continue
        if candidates:
            candidates.sort(key=lambda item: (item.rectangle().top, item.rectangle().left))
            first_row_top = candidates[0].rectangle().top
            first_row = [item for item in candidates if abs(item.rectangle().top - first_row_top) <= 8]
            first_row.sort(key=lambda item: item.rectangle().left)
            if first_row:
                return first_row
        time.sleep(0.2)
    raise TimeoutError("Checkboxes de inclusao no salvamento nao encontrados.")


def mark_all_images_for_saving(
    main_window,
    robot: HumanRobot,
    total_images: int,
) -> int:
    """Percorre as imagens pelo atalho Tk e garante sua inclusao no salvamento."""
    main_window.set_focus()
    for index in range(1, int(total_images) + 1):
        robot.keys(
            "{F12}",
            f"Marcar imagem {index}/{total_images} para salvamento",
            delay_after=SAVE_SELECTION_DELAY_SECONDS,
        )
    return int(total_images)


def find_editor_window(application):
    try:
        windows = application.windows()
    except Exception:
        return None
    for window in windows:
        try:
            if window.is_visible() and _normalize(window.window_text()).startswith("figure"):
                return window
        except Exception:
            continue
    return None


def edit_every_image(
    application,
    robot: HumanRobot,
    maximum_images: int,
    open_timeout: float = 120.0,
) -> int:
    deadline = time.monotonic() + open_timeout
    editor = None
    while time.monotonic() < deadline:
        editor = find_editor_window(application)
        if editor is not None:
            break
        dialogs = visible_dialogs(application)
        if dialogs:
            answer_dialog(robot, dialogs[0], prefer_yes=True)
        time.sleep(0.2)
    if editor is None:
        raise TimeoutError("A janela do editor de mascaras nao apareceu.")

    accepted = 0
    while accepted < maximum_images:
        editor = find_editor_window(application)
        if editor is None:
            return accepted
        image_number = accepted + 1
        editor.set_focus()
        robot.wait_for_image_edit(image_number)
        editor = find_editor_window(application)
        if editor is None:
            return accepted
        editor.set_focus()
        accepted += 1
        robot.keys("{ENTER}", f"Salvar mascara e avancar ({accepted})")

    if find_editor_window(application) is not None:
        raise RuntimeError(
            "O editor ainda esta aberto apos confirmar todas as imagens encontradas na pasta."
        )
    return accepted


def wait_for_robot_edit_completion(
    application,
    main_window,
    robot: HumanRobot,
    expected_images: int,
    timeout: float,
) -> None:
    """Espera a aplicacao das ROIs e valida a confirmacao publicada pelo CORA."""
    marker_re = re.compile(r"\[ROBOT_EDIT_DONE:(\d+)/(\d+)\]")
    deadline = time.monotonic() + max(1.0, float(timeout))
    while time.monotonic() < deadline:
        dismiss_information_dialogs(application, robot)
        try:
            title = str(main_window.window_text())
        except Exception:
            title = ""
        match = marker_re.search(title)
        if match is not None:
            applied, requested = (int(value) for value in match.groups())
            if requested != int(expected_images) or applied != int(expected_images):
                raise RuntimeError(
                    "Nem todas as mascaras foram aplicadas apos a edicao: "
                    f"{applied}/{requested}; esperado {expected_images}."
                )
            print(f"Mascaras aplicadas e reprocessadas: {applied}/{requested}", flush=True)
            return
        time.sleep(0.25)
    raise TimeoutError("O CORA nao confirmou o reprocessamento das mascaras editadas.")


def wait_for_saved_files(output_dir: Path, started_epoch: float, timeout: float = 120.0) -> None:
    summary = output_dir / "resumo_resultados.csv"
    page_times = output_dir / "tempos_por_pagina.csv"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if summary.is_file() and page_times.is_file():
            if min(summary.stat().st_mtime, page_times.stat().st_mtime) >= started_epoch - 1.0:
                return
        time.sleep(0.25)
    raise TimeoutError("Os CSVs finais nao foram gerados no tempo esperado.")


def validate_saved_results(
    images_dir: Path,
    output_dir: Path,
    test_number: int,
) -> tuple[int, int]:
    source_images = {
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    summary_path = output_dir / "resumo_resultados.csv"
    with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))

    saved_sources: set[Path] = set()
    missing_overlays: list[str] = []
    processing_errors: list[str] = []
    for row in rows:
        source_text = str(row.get("arquivo_origem", "")).strip()
        if source_text:
            saved_sources.add(Path(source_text).resolve())
        overlay_name = str(row.get("overlay_png", "")).strip()
        if not overlay_name or not (output_dir / overlay_name).is_file():
            missing_overlays.append(source_text or "<origem desconhecida>")
        error = str(row.get("erro", "")).strip()
        if error:
            processing_errors.append(f"{source_text}: {error}")

    missing_sources = sorted(str(path) for path in source_images - saved_sources)
    if missing_sources:
        preview = "\n".join(missing_sources[:20])
        raise RuntimeError(f"Imagens da pasta ausentes no resultado:\n{preview}")
    if missing_overlays:
        preview = "\n".join(missing_overlays[:20])
        raise RuntimeError(f"Imagens sem overlay salvo:\n{preview}")
    if processing_errors:
        preview = "\n".join(processing_errors[:20])
        raise RuntimeError(f"Ocorreram erros de processamento:\n{preview}")

    page_times_path = output_dir / "tempos_por_pagina.csv"
    with page_times_path.open("r", newline="", encoding="utf-8-sig") as handle:
        timing_rows = list(csv.DictReader(handle, delimiter=";"))
    timing_pages = {str(row.get("pagina", "")) for row in timing_rows}
    if "edicao_de_mascara" not in timing_pages:
        raise RuntimeError("O CSV de tempos nao contem a pagina de edicao de mascara.")
    timing_by_page = {str(row.get("pagina", "")): row for row in timing_rows}
    processing_wait = timing_by_page.get("espera_processamento_imagem")
    if processing_wait is None:
        raise RuntimeError("O CSV de tempos nao separou a espera do processamento.")
    try:
        processing_visits = int(processing_wait.get("numero_de_visitas", 0) or 0)
        processing_seconds = float(processing_wait.get("tempo_total_segundos", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Tempo de espera do processamento invalido no CSV.") from exc
    if processing_visits < 1 or processing_seconds <= 0.0:
        raise RuntimeError("A espera do processamento nao foi medida separadamente.")
    if int(test_number) == 2:
        if processing_visits < 2:
            raise RuntimeError(
                "O Teste 2 nao registrou separadamente o processamento inicial e o reprocessamento."
            )
        editor_time = timing_by_page.get("edicao_de_mascara")
        try:
            editor_visits = int((editor_time or {}).get("numero_de_visitas", 0) or 0)
            editor_seconds = float((editor_time or {}).get("tempo_total_segundos", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Tempo do editor de mascaras invalido no CSV.") from exc
        if editor_visits < 1 or editor_seconds <= 0.0:
            raise RuntimeError("O Teste 2 nao mediu o tempo de edicao das mascaras.")

    return len(source_images), len(rows)


def write_action_log(
    output_dir: Path,
    records: list[ActionRecord],
    status: str,
    test_number: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"teste_robotizado_{test_number}_acoes.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["data_hora", "acao"], delimiter=";")
        writer.writeheader()
        for record in records:
            writer.writerow({"data_hora": record.timestamp, "acao": record.action})
        writer.writerow(
            {
                "data_hora": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "acao": f"STATUS FINAL: {status}",
            }
        )
    return path


def parse_args(test_number: int = 2) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Teste robotizado {test_number} do CORA com pywinauto."
    )
    parser.add_argument(
        "--timeout-processamento",
        type=float,
        default=3600.0,
        help="Limite em segundos para cada etapa de processamento.",
    )
    parser.add_argument(
        "--fechar-ao-final",
        action="store_true",
        help="Fecha a interface depois da validacao; por padrao ela permanece aberta.",
    )
    return parser.parse_args()


def run_robotized_test(
    test_number: int,
    interval_description: str,
    execute_specific_flow: Callable[[RobotizedTestContext], None],
) -> int:
    if test_number not in (1, 2):
        raise ValueError(f"Numero de teste nao suportado: {test_number}")
    args = parse_args(test_number=test_number)
    ensure_pywinauto_interpreter()
    configure_comtypes_cache()
    images_dir = ROBOT_TEST_FOLDER.resolve()
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Pasta fixa dos testes nao encontrada: {images_dir}")

    source_images = [
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not source_images:
        raise RuntimeError(f"Nenhuma imagem suportada encontrada em: {images_dir}")

    # PT: Mantem os arquivos gerados dentro da pasta fixa, mas em um subdiretorio para que overlays de execucoes anteriores nao sejam lidos como novas entradas.
    # EN: Keeps generated files inside the fixed folder but in a subdirectory so overlays from previous runs are not read as new inputs.
    output_dir = images_dir / "_cora_resultados"
    output_dir.mkdir(parents=True, exist_ok=True)

    dpi, mouse_speed = configure_windows_dpi()
    print(
        f"Teste robotizado {test_number} | "
        f"pasta fixa: {images_dir} | "
        f"DPI do sistema: {dpi if dpi is not None else 'indisponivel'} | "
        f"velocidade atual do mouse: {mouse_speed if mouse_speed is not None else 'indisponivel'} | "
        f"pausa por acao: {ACTION_DELAY_SECONDS:.1f}s | "
        f"{interval_description}",
        flush=True,
    )

    try:
        from pywinauto import Application, Desktop, keyboard
    except ImportError as exc:
        requirements_path = Path(__file__).resolve().parent / "requirements_robotizado.txt"
        raise RuntimeError(
            "pywinauto nao esta instalado no Python atual. Execute: "
            f"{sys.executable} -m pip install -r {requirements_path}"
        ) from exc

    robot = HumanRobot(keyboard)
    application = None
    main_window = None
    status = "falha antes da inicializacao"
    run_started_epoch = time.time()

    try:
        matching_windows = Desktop(backend="win32").windows(
            title_re=MAIN_WINDOW_RE,
            visible_only=True,
        )
        if not matching_windows:
            raise RuntimeError(
                "A interface do CORA nao esta aberta. Abra a tela inicial do programa "
                "antes de executar este arquivo no VS Code."
            )
        if len(matching_windows) > 1:
            raise RuntimeError(
                "Mais de uma interface do CORA esta aberta. Deixe somente uma instancia ativa."
            )

        main_handle = matching_windows[0].handle
        application = Application(backend="win32").connect(handle=main_handle)
        main_window = application.window(handle=main_handle)
        main_window.wait("visible enabled ready", timeout=60)
        robot.maximize(main_window)

        open_robot_test_page(main_window, robot)

        set_robot_field_from_clipboard(
            main_window,
            robot,
            "{F6}",
            str(images_dir),
            "pasta de imagens",
            "ROBOT_IMAGES_OK",
        )
        set_robot_field_from_clipboard(
            main_window,
            robot,
            "{F7}",
            str(output_dir),
            "pasta de saida",
            "ROBOT_PATHS_OK",
        )

        main_window.set_focus()
        robot.keys("{F8}", "Carregar os grupos de imagens")
        review_deadline = time.monotonic() + 120.0
        while not progress_window_visible(application):
            if time.monotonic() >= review_deadline:
                raise TimeoutError("A revisao dos grupos nao iniciou o processamento.")
            main_window.set_focus()
            robot.keys("{F9}", "Marcar todas as imagens para processamento")
            robot.keys("{F10}", "Preencher automaticamente tempos vazios")
            robot.keys("{F11}", "Confirmar a revisao e iniciar o processamento")

        wait_for_processing_cycle(
            application,
            main_window,
            robot,
            timeout=max(30.0, float(args.timeout_processamento)),
        )

        execute_specific_flow(
            RobotizedTestContext(
                application=application,
                main_window=main_window,
                robot=robot,
                source_images=source_images,
                processing_timeout=max(30.0, float(args.timeout_processamento)),
            )
        )

        main_window.set_focus()
        robot.keys("+{F12}", "Salvar todos os resultados")
        wait_for_saved_files(output_dir, run_started_epoch, timeout=180)
        dialogs = visible_dialogs(application)
        if dialogs:
            answer_dialog(robot, dialogs[0], prefer_yes=False)

        source_count, result_count = validate_saved_results(
            images_dir,
            output_dir,
            test_number=test_number,
        )
        status = f"sucesso: {source_count} imagens de origem; {result_count} resultados salvos"
        print(status, flush=True)

        if args.fechar_ao_final:
            main_window.set_focus()
            robot.keys("%{F4}", "Fechar o CORA")
        return 0
    except Exception as exc:
        status = f"falha: {type(exc).__name__}: {exc}"
        print(status, file=sys.stderr, flush=True)
        raise
    finally:
        log_path = write_action_log(output_dir, robot.records, status, test_number=test_number)
        print(f"Log de acoes: {log_path}", flush=True)
