"""Tela inicial para escolher o modo de execucao do CORA."""

import tkinter as tk
from tkinter import ttk


def build_test_selection_layout(app) -> None:
    """Monta a tela exibida antes das configuracoes e do visualizador."""
    page = ttk.Frame(app.root, padding=28)
    app.test_selection_page = page

    header = ttk.Frame(page)
    header.pack(fill=tk.X, pady=(18, 24))
    ttk.Label(header, text="CORA - Cell Open-Region Analyzer", font=("TkDefaultFont", 24, "bold")).pack(anchor=tk.W)
    ttk.Label(
        header,
        text=app._t("test.title"),
        font=("TkDefaultFont", 13),
    ).pack(anchor=tk.W, pady=(6, 0))

    cards = ttk.Frame(page)
    cards.pack(fill=tk.X, anchor=tk.N)
    cards.columnconfigure(0, weight=1, uniform="test_mode")
    cards.columnconfigure(1, weight=1, uniform="test_mode")
    cards.columnconfigure(2, weight=1, uniform="test_mode")

    single_card = ttk.LabelFrame(cards, text=app._t("test.single_title"), padding=20)
    single_card.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10))
    ttk.Label(
        single_card,
        text=app._t("test.single_description"),
        justify=tk.LEFT,
        wraplength=470,
    ).pack(fill=tk.X, anchor=tk.W)
    app.single_image_test_btn = ttk.Button(
        single_card,
        text=app._t("button.choose_single_image"),
        command=app._choose_single_image_test,
        padding=(14, 9),
    )
    app.single_image_test_btn.pack(fill=tk.X, pady=(20, 0))

    batch_card = ttk.LabelFrame(cards, text=app._t("test.batch_title"), padding=20)
    batch_card.grid(row=0, column=1, sticky=tk.NSEW, padx=(10, 0))
    ttk.Label(
        batch_card,
        text=app._t("test.batch_description"),
        justify=tk.LEFT,
        wraplength=470,
    ).pack(fill=tk.X, anchor=tk.W)
    app.batch_mode_btn = ttk.Button(
        batch_card,
        text=app._t("button.open_batch"),
        command=app._open_normal_batch_page,
        padding=(14, 9),
    )
    app.batch_mode_btn.pack(fill=tk.X, pady=(20, 0))

    batch_test_card = ttk.LabelFrame(cards, text=app._t("test.batch_test_title"), padding=20)
    batch_test_card.grid(row=0, column=2, sticky=tk.NSEW, padx=(10, 0))
    ttk.Label(
        batch_test_card,
        text=app._t("test.batch_test_description"),
        justify=tk.LEFT,
        wraplength=360,
    ).pack(fill=tk.X, anchor=tk.W)

    app.batch_test_buttons = []
    for batch_size in (10, 24, 60):
        button = ttk.Button(
            batch_test_card,
            text=app._t("button.start_batch_test", count=batch_size),
            command=lambda size=batch_size: app._choose_batch_performance_test(size),
            padding=(14, 7),
        )
        button.pack(fill=tk.X, pady=(10 if batch_size == 10 else 6, 0))
        app.batch_test_buttons.append(button)

    robotized_card = ttk.LabelFrame(cards, text=app._t("test.robotized_title"), padding=20)
    robotized_card.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(20, 0))
    robotized_card.columnconfigure(0, weight=1)
    ttk.Label(
        robotized_card,
        text=app._t("test.robotized_description"),
        justify=tk.LEFT,
        wraplength=760,
    ).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
    app.robotized_test_btn = ttk.Button(
        robotized_card,
        text=app._t("button.open_robotized_test"),
        command=app._open_robot_test_shortcut,
        padding=(14, 9),
    )
    app.robotized_test_btn.grid(row=0, column=1, sticky=tk.E)

    ttk.Label(
        page,
        text=app._t("test.extensible_hint"),
    ).pack(anchor=tk.W, pady=(24, 0))

    if app.config_page is not None and app.config_page.winfo_manager():
        app.config_page.pack_forget()
    page.pack(fill=tk.BOTH, expand=True)
