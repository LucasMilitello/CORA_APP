"""Coleta e persistencia das metricas do modo de teste de imagem unica."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import os
import threading

try:
    import psutil
except ImportError:  # PT: O app continua funcional e informa metricas indisponiveis. | EN: The app remains functional and reports unavailable metrics.
    psutil = None


@dataclass(frozen=True)
class ResourcePeaks:
    """Maiores valores observados durante uma execucao."""

    ram_mb: float | None
    cpu_percent: float | None


class ProcessResourceMonitor:
    """Amostra RAM residente e CPU do processo sem bloquear o processamento."""

    def __init__(self, sample_interval_s: float = 0.10) -> None:
        self.sample_interval_s = max(0.05, float(sample_interval_s))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_ram_mb: float | None = None
        self._peak_cpu_percent: float | None = None

    @property
    def available(self) -> bool:
        return psutil is not None

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourcePeaks:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.5, self.sample_interval_s * 3.0))
        self._thread = None
        return ResourcePeaks(self._peak_ram_mb, self._peak_cpu_percent)

    def _sample_loop(self) -> None:
        if psutil is None:
            return
        process = psutil.Process(os.getpid())
        logical_cpus = max(1, int(psutil.cpu_count(logical=True) or 1))
        try:
            process.cpu_percent(interval=None)
            while not self._stop_event.wait(self.sample_interval_s):
                ram_mb = float(process.memory_info().rss) / (1024.0 * 1024.0)
                # PT: psutil pode retornar N * 100 para processos multithread. | EN: psutil may return N * 100 for multithreaded processes.
                # PT: Normalizamos pela quantidade de CPUs para a escala do sistema (0-100%). | EN: Normalize by the CPU count to obtain the system-wide scale (0-100%).
                cpu_percent = min(100.0, float(process.cpu_percent(interval=None)) / logical_cpus)
                self._peak_ram_mb = max(self._peak_ram_mb or ram_mb, ram_mb)
                self._peak_cpu_percent = max(self._peak_cpu_percent or cpu_percent, cpu_percent)
            ram_mb = float(process.memory_info().rss) / (1024.0 * 1024.0)
            cpu_percent = min(100.0, float(process.cpu_percent(interval=None)) / logical_cpus)
            self._peak_ram_mb = max(self._peak_ram_mb or ram_mb, ram_mb)
            self._peak_cpu_percent = max(self._peak_cpu_percent or cpu_percent, cpu_percent)
        except (psutil.Error, OSError):
            return


TEST_REPORT_FIELDS = (
    "executado_em",
    "repeticao",
    "imagem",
    "caminho_imagem",
    "sucesso",
    "cancelado",
    "tempo_carregamento_s",
    "tempo_segmentacao_s",
    "tempo_total_pipeline_s",
    "ram_maxima_mb",
    "cpu_maxima_percent",
    "area_detectada_px",
    "largura_px",
    "altura_px",
    "erro",
)

BATCH_TEST_REPORT_FIELDS = (
    "executado_em",
    "tamanho_lote",
    "iteracao",
    "imagens_planejadas",
    "imagens_processadas",
    "sucessos",
    "falhas",
    "tempo_carregamento_s",
    "tempo_segmentacao_s",
    "tempo_total_s",
    "ram_maxima_mb",
    "cpu_maxima_percent",
    "imagens_por_segundo",
    "cancelado",
    "erro",
)

CSV_DELIMITER = ";"


def _detect_csv_delimiter(sample: str) -> str:
    """Detecta relatorios antigos, que usavam a virgula como separador."""
    header = sample.splitlines()[0] if sample else ""
    candidates = (CSV_DELIMITER, ",", "\t")
    delimiter = max(candidates, key=header.count)
    return delimiter if header.count(delimiter) else ","


def append_test_report(csv_path: Path, row: dict[str, object]) -> Path:
    """Acrescenta uma execucao ao CSV, criando cabecalho quando necessario."""
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0
    if not write_header:
        with target.open("r", newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            previous_delimiter = _detect_csv_delimiter(sample)
            reader = csv.DictReader(handle, delimiter=previous_delimiter)
            previous_fields = tuple(reader.fieldnames or ())
            if previous_fields != TEST_REPORT_FIELDS or previous_delimiter != CSV_DELIMITER:
                previous_rows = list(reader)
            else:
                previous_rows = None
        if previous_rows is not None:
            # PT: Migra relatorios antigos (cabecalho ou delimitador) sem perder dados. | EN: Migrates older reports (header or delimiter) without losing data.
            with target.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=TEST_REPORT_FIELDS,
                    delimiter=CSV_DELIMITER,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(
                    {field: previous_row.get(field, "") for field in TEST_REPORT_FIELDS}
                    for previous_row in previous_rows
                )
    with target.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TEST_REPORT_FIELDS,
            delimiter=CSV_DELIMITER,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in TEST_REPORT_FIELDS})
    return target


def read_test_report(csv_path: Path) -> list[dict[str, str]]:
    """Le todas as rodadas do relatorio, inclusive arquivos no formato antigo."""
    target = Path(csv_path)
    if not target.exists() or target.stat().st_size == 0:
        return []
    with target.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=_detect_csv_delimiter(sample))
        return [
            {field: row.get(field, "") for field in TEST_REPORT_FIELDS}
            for row in reader
        ]


def append_batch_test_report(csv_path: Path, row: dict[str, object]) -> Path:
    """Acrescenta uma iteracao completa do teste de batching ao CSV."""
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0
    if not write_header:
        with target.open("r", newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            previous_delimiter = _detect_csv_delimiter(sample)
            reader = csv.DictReader(handle, delimiter=previous_delimiter)
            previous_fields = tuple(reader.fieldnames or ())
            previous_rows = (
                list(reader)
                if previous_fields != BATCH_TEST_REPORT_FIELDS or previous_delimiter != CSV_DELIMITER
                else None
            )
        if previous_rows is not None:
            with target.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=BATCH_TEST_REPORT_FIELDS,
                    delimiter=CSV_DELIMITER,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(
                    {field: previous_row.get(field, "") for field in BATCH_TEST_REPORT_FIELDS}
                    for previous_row in previous_rows
                )
    with target.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=BATCH_TEST_REPORT_FIELDS,
            delimiter=CSV_DELIMITER,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in BATCH_TEST_REPORT_FIELDS})
    return target


def read_batch_test_report(csv_path: Path) -> list[dict[str, str]]:
    """Le o historico de iteracoes do teste de batching."""
    target = Path(csv_path)
    if not target.exists() or target.stat().st_size == 0:
        return []
    with target.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=_detect_csv_delimiter(sample))
        return [
            {field: row.get(field, "") for field in BATCH_TEST_REPORT_FIELDS}
            for row in reader
        ]


__all__ = [
    "ProcessResourceMonitor",
    "ResourcePeaks",
    "append_test_report",
    "read_test_report",
    "append_batch_test_report",
    "read_batch_test_report",
]
