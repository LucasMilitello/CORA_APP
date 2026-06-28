"""Construcao do layout principal da GUI com duas telas: Configuracoes e Visualizacao."""

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def build_main_layout(app, time_order: tuple[str, ...]) -> None:
    """Monta os widgets principais e conecta eventos com a classe CORAApp."""
    button_padding = (10, 4)

    # =========================
    # PT: Tela 1: Configuracoes | EN: Screen 1: Settings
    # =========================
    config_page = ttk.Frame(app.root, padding=14)
    config_page.pack(fill=tk.BOTH, expand=True)
    app.config_page = config_page

    config_card = ttk.Frame(config_page, padding=10)
    config_card.pack(fill=tk.X, anchor=tk.N)

    app.config_guide_btn = ttk.Button(
        config_card,
        text="Guia",
        command=app._open_guide_popup,
        padding=button_padding,
    )
    app.config_guide_btn.grid(row=0, column=0, sticky=tk.W)
    ttk.Label(config_card, text="Configuracoes iniciais").grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
    app.theme_toggle_btn_config = ttk.Button(
        config_card,
        text=app._ui_theme_toggle_label(),
        command=app._open_theme_settings_popup,
        padding=button_padding,
    )
    app.theme_toggle_btn_config.grid(row=0, column=2, sticky=tk.E)
    app.language_toggle_btn_config = ttk.Button(
        config_card,
        text=app._ui_language_toggle_label(),
        command=app._toggle_ui_language,
        padding=button_padding,
    )
    app.language_toggle_btn_config.grid(row=0, column=3, sticky=tk.E, padx=(8, 0))

    app.test_modes_btn_config = ttk.Button(
        config_card,
        text=app._t("button.test_modes"),
        command=app._show_test_selection_page,
        padding=button_padding,
    )
    app.test_modes_btn_config.grid(row=3, column=0, columnspan=4, sticky=tk.EW, pady=(8, 0))

    ttk.Label(config_card, text="Pasta de imagens:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
    app.folder_entry = ttk.Entry(config_card, textvariable=app.folder_var)
    app.folder_entry.grid(row=1, column=1, sticky=tk.EW, padx=6, pady=(10, 0))
    app.browse_btn = ttk.Button(
        config_card,
        text="Procurar pasta",
        command=app._browse_folder,
        padding=button_padding,
    )
    app.browse_btn.grid(row=1, column=2, padx=4, pady=(10, 0), sticky=tk.EW)

    ttk.Label(config_card, text="Pasta de saida (opcional):").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
    app.output_folder_entry = ttk.Entry(config_card, textvariable=app.output_folder_var)
    app.output_folder_entry.grid(row=2, column=1, sticky=tk.EW, padx=6, pady=(8, 0))
    app.browse_output_btn = ttk.Button(
        config_card,
        text="Escolher saida",
        command=app._browse_output_folder,
        padding=button_padding,
    )
    app.browse_output_btn.grid(row=2, column=2, padx=4, pady=(8, 0), sticky=tk.EW)

    app.scan_btn = ttk.Button(
        config_card,
        text="Carregar grupos",
        command=app._scan_folder,
        padding=button_padding,
    )
    app.scan_btn.grid(row=4, column=0, columnspan=4, sticky=tk.EW, pady=(12, 0))

    config_card.columnconfigure(1, weight=1)

    # PT: Espaco livre da tela inicial reservado para revisao do agrupamento. | EN: Free space on the initial screen reserved for reviewing the grouping.
    review_host = ttk.Frame(config_page, padding=(0, 12, 0, 0))
    review_host.pack(fill=tk.BOTH, expand=True)
    app.group_review_host = review_host
    # ========================
    # PT: Tela 2: Visualizacao | EN: Screen 2: Viewer
    # ========================
    viewer_page = ttk.Frame(app.root)
    app.viewer_page = viewer_page

    top = ttk.Frame(viewer_page, padding=10)
    top.pack(fill=tk.X)

    # PT: Menu de configuracoes (estilo File) no canto esquerdo da tela de visualizacao. | EN: File-style settings menu on the left side of the viewer screen.
    app.open_config_btn = ttk.Menubutton(
        top,
        text="Configuracoes v",
        direction="below",
        padding=button_padding,
    )
    app.viewer_guide_btn = ttk.Button(
        top,
        text="Guia",
        command=app._open_guide_popup,
        padding=button_padding,
    )
    app.viewer_guide_btn.grid(row=0, column=0, sticky=tk.W)
    app.open_config_btn.grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
    config_menu = tk.Menu(app.open_config_btn, tearoff=False)
    app.open_config_menu = config_menu
    config_menu.add_command(label="Pastas de entrada/saida", command=app._open_paths_mini_tab)
    config_menu.add_command(label="Navegacao de grupos", command=app._open_group_navigation_mini_tab)
    config_menu.add_command(label="Cores dos contornos", command=app._open_contour_settings_popup)
    config_menu.add_separator()
    config_menu.add_command(label="Abrir configuracoes completas", command=app._show_config_page)
    config_menu.add_command(label=app._ui_language_toggle_label(), command=app._toggle_ui_language)
    app.language_toggle_menu_index = 5
    config_menu.add_command(label=app._t("test.menu_choose"), command=app._show_test_selection_page)
    app.open_config_btn.configure(menu=config_menu)
    app.theme_toggle_btn = ttk.Button(
        top,
        text=app._ui_theme_toggle_label(),
        command=app._open_theme_settings_popup,
        padding=button_padding,
    )
    app.theme_toggle_btn.grid(row=0, column=2, sticky=tk.E)
    app._refresh_theme_toggle_buttons()

    ttk.Label(top, text="Grupo em revisao:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
    app.group_combo = ttk.Combobox(top, textvariable=app.group_var, state="disabled")
    app.group_combo.grid(row=1, column=1, sticky=tk.EW, padx=(8, 6), pady=(8, 0))
    app.group_combo.bind("<<ComboboxSelected>>", app._on_group_selected)

    nav = ttk.Frame(top)
    nav.grid(row=1, column=2, sticky=tk.EW, pady=(8, 0))
    app.toggle_results_btn = ttk.Button(
        nav,
        text="Ocultar resultados",
        command=app._toggle_results_panel,
        padding=button_padding,
    )
    app.toggle_results_btn.grid(row=0, column=0, sticky=tk.EW)
    nav.columnconfigure(0, weight=1)

    mark = ttk.Frame(top)
    mark.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))

    ttk.Label(mark, text="Incluir no salvamento:").grid(row=0, column=0, sticky=tk.W)
    for idx, time_tag in enumerate(time_order, start=1):
        var = tk.BooleanVar(value=False)
        check = ttk.Checkbutton(mark, text=time_tag.upper(), variable=var, command=app._on_mark_changed)
        check.grid(row=0, column=idx, padx=(8, 0), sticky=tk.W)
        app.refazer_vars[time_tag] = var
        app.refazer_checks[time_tag] = check

    ttk.Label(mark, text="Sem area:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
    for idx, time_tag in enumerate(time_order, start=1):
        var = tk.BooleanVar(value=False)
        check = ttk.Checkbutton(mark, text=time_tag.upper(), variable=var, command=app._on_mark_changed)
        check.grid(row=1, column=idx, padx=(8, 0), pady=(6, 0), sticky=tk.W)
        app.no_area_vars[time_tag] = var
        app.no_area_checks[time_tag] = check

    nav_buttons = ttk.Frame(mark)
    nav_buttons.grid(row=0, column=len(time_order) + 1, rowspan=2, sticky=tk.N, padx=(16, 0))
    app.prev_btn = ttk.Button(
        nav_buttons,
        text="Grupo anterior",
        command=app._show_prev_group,
        padding=button_padding,
    )
    app.prev_btn.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
    app.next_btn = ttk.Button(
        nav_buttons,
        text="Proximo grupo",
        command=app._show_next_group,
        padding=button_padding,
    )
    app.next_btn.grid(row=0, column=1, sticky=tk.EW, padx=(4, 0))
    nav_buttons.columnconfigure(0, weight=1)
    nav_buttons.columnconfigure(1, weight=1)

    actions = ttk.Frame(mark)
    actions.grid(row=0, column=len(time_order) + 2, rowspan=2, sticky=tk.EW, padx=(16, 0))

    app.redefine_roi_btn = ttk.Button(
        actions,
        text="Editar mascaras",
        command=app._redefine_roi_for_unselected,
        padding=button_padding,
    )
    app.redefine_roi_btn.grid(row=0, column=0, sticky=tk.EW, padx=(0, 4), pady=(0, 4))

    app.clear_roi_btn = ttk.Button(
        actions,
        text="Restaurar auto",
        command=app._clear_roi_for_unselected,
        padding=button_padding,
    )
    app.clear_roi_btn.grid(row=0, column=1, sticky=tk.EW, padx=(4, 0), pady=(0, 4))

    app.save_btn = ttk.Button(
        actions,
        text="Salvar resultado",
        command=app._save_results_clicked,
        padding=button_padding,
    )
    app.save_btn.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(2, 0))
    actions.columnconfigure(0, weight=1)
    actions.columnconfigure(1, weight=1)

    mark.columnconfigure(len(time_order) + 2, weight=1)

    top.columnconfigure(0, weight=0)
    top.columnconfigure(1, weight=0)
    top.columnconfigure(2, weight=1, minsize=220)

    body = ttk.Panedwindow(viewer_page, orient=tk.HORIZONTAL)
    body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
    app.body_paned = body

    left = ttk.Frame(body, padding=8)
    right = ttk.Frame(body, padding=8)
    app.left_panel = left
    app.right_panel = right
    body.add(left, weight=2)
    body.add(right, weight=5)

    left_header = ttk.Frame(left)
    left_header.pack(fill=tk.X)
    ttk.Label(left_header, text="Resultados do grupo").pack(side=tk.LEFT, anchor=tk.W)
    app.side_notebook = ttk.Notebook(left)
    app.side_notebook.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    app.metrics_tab = ttk.Frame(app.side_notebook, padding=0)
    app.progress_tab = None
    app.side_notebook.add(app.metrics_tab, text="Resultados")

    app.metrics_text = tk.Text(app.metrics_tab, width=46, height=30, state=tk.DISABLED)
    app.metrics_text.pack(fill=tk.BOTH, expand=True)

    app.inline_progress_frame = ttk.Frame(app.metrics_tab)
    ttk.Label(app.inline_progress_frame, text="Andamento do processamento (popup fechado)").pack(anchor=tk.W)
    info_grid = ttk.Frame(app.inline_progress_frame)
    info_grid.pack(fill=tk.X, pady=(6, 6))
    ttk.Label(info_grid, text="Imagem atual:").grid(row=0, column=0, sticky=tk.W)
    ttk.Label(
        info_grid,
        textvariable=app.progress_current_image_var,
        anchor=tk.W,
        justify=tk.LEFT,
        wraplength=310,
    ).grid(row=0, column=1, sticky=tk.W, padx=(6, 0))
    ttk.Label(info_grid, text="Faltando:").grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
    ttk.Label(info_grid, textvariable=app.progress_remaining_var, anchor=tk.W).grid(
        row=1, column=1, sticky=tk.W, padx=(6, 0), pady=(4, 0)
    )
    ttk.Label(info_grid, text="ETA:").grid(row=2, column=0, sticky=tk.W, pady=(4, 0))
    ttk.Label(info_grid, textvariable=app.progress_eta_var, anchor=tk.W).grid(
        row=2, column=1, sticky=tk.W, padx=(6, 0), pady=(4, 0)
    )
    info_grid.columnconfigure(1, weight=1)
    progress_row = ttk.Frame(app.inline_progress_frame)
    progress_row.pack(fill=tk.X, pady=(0, 6))
    ttk.Progressbar(
        progress_row,
        mode="determinate",
        variable=app.progress_var,
        maximum=100.0,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Label(progress_row, textvariable=app.progress_percent_var, width=8, anchor=tk.E).pack(side=tk.LEFT, padx=(8, 0))
    app.cancel_btn = ttk.Button(
        app.inline_progress_frame,
        text="Cancelar processamento",
        command=app._request_cancel_processing,
        state=tk.DISABLED,
        padding=button_padding,
    )
    app.cancel_btn.pack(anchor=tk.E, pady=(2, 0))
    app.inline_progress_frame.pack(fill=tk.X, pady=(0, 6), before=app.metrics_text)
    app.inline_progress_frame.pack_forget()
    app._refresh_help_tab()

    right_header = ttk.Frame(right)
    right_header.pack(fill=tk.X)
    ttk.Label(right_header, text="Imagens do grupo").pack(side=tk.LEFT, anchor=tk.W)
    app.contour_settings_btn = ttk.Button(
        right_header,
        text="Configurar contornos",
        command=app._open_contour_settings_popup,
        padding=button_padding,
    )
    app.contour_settings_btn.pack(side=tk.RIGHT, anchor=tk.E)
    app.compare_area_overlay_check = ttk.Checkbutton(
        right_header,
        text=app._t("button.compare_areas"),
        variable=app.compare_area_overlay_var,
        command=app._on_compare_area_overlay_changed,
    )
    app.compare_area_overlay_check.pack(side=tk.RIGHT, anchor=tk.E, padx=(0, 10))

    app.figure = Figure(figsize=(11.4, 6.9), dpi=100)
    app.canvas = FigureCanvasTkAgg(app.figure, master=right)
    app.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    bottom = ttk.Frame(viewer_page, padding=(10, 0, 10, 8))
    bottom.pack(side=tk.BOTTOM, fill=tk.X)
    app.viewer_bottom = bottom
    ttk.Label(bottom, textvariable=app.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(4, 0))

    # PT: Inicia na tela de configuracoes. | EN: Starts on the settings screen.
    app._show_config_page(set_status=False)
