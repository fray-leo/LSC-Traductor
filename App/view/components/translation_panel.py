"""
view/components/translation_panel.py

Panel derecho de la interfaz. Muestra:
  - La seña detectada actualmente en texto grande.
  - Una barra de confianza del modelo.
  - El historial scrollable de las últimas traducciones.
  - Un botón para limpiar el historial.

No accede a la base de datos directamente — recibe los datos
desde main.py a través de update_translation() y update_history().
"""

import tkinter as tk
from tkinter import font as tkfont


# Paleta — compatible con fondos oscuros y claros
_BG         = "#1e1e1e"
_BG_PANEL   = "#2a2a2a"
_BG_ITEM    = "#333333"
_FG_PRIMARY = "#ffffff"
_FG_MUTED   = "#888888"
_FG_LABEL   = "#aaaaaa"
_ACCENT     = "#4a9eff"
_UNCERTAIN  = "#555555"
_BAR_EMPTY  = "#3a3a3a"
_BAR_LOW    = "#e05c5c"
_BAR_MID    = "#e0a840"
_BAR_HIGH   = "#4caf72"
_BORDER     = "#3d3d3d"


class TranslationPanel(tk.Frame):
    """
    Panel de traducción con seña actual, barra de confianza e historial.

    Uso desde app.py:
        panel = TranslationPanel(parent, on_clear=db.clear)
        panel.pack(fill="both", expand=True)
        panel.update_translation("hola", confidence=0.92)
        panel.update_history([("hola", "12:01:05"), ("gracias", "12:01:12")])
    """

    UNCERTAIN_LABEL = "..."
    MAX_HISTORY_ITEMS = 10

    def __init__(self, parent: tk.Widget, on_clear=None, **kwargs):
        """
        Args:
            parent:   widget padre de tkinter.
            on_clear: callable opcional que se llama al pulsar "Limpiar historial".
                      Debe encargarse de borrar la base de datos; este widget
                      solo limpia su propia lista visual.
        """
        super().__init__(parent, bg=_BG, **kwargs)
        self._on_clear = on_clear
        self._history_items: list[tk.Widget] = []

        self._build()

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def update_translation(self, sena: str, confidence: float = 0.0):
        """
        Actualiza la seña mostrada y la barra de confianza.

        Args:
            sena:       nombre de la seña detectada o "..." si incierta.
            confidence: valor entre 0.0 y 1.0 del modelo.
        """
        is_uncertain = (sena == self.UNCERTAIN_LABEL or not sena)

        # Texto de la seña
        display = sena.replace("_", " ").upper() if not is_uncertain else "—"
        color = _FG_PRIMARY if not is_uncertain else _UNCERTAIN
        self._sign_label.configure(text=display, fg=color)

        # Subtítulo de confianza
        if is_uncertain:
            self._conf_text.configure(text="esperando seña...", fg=_FG_MUTED)
        else:
            pct = int(confidence * 100)
            self._conf_text.configure(text=f"confianza  {pct}%", fg=_FG_LABEL)

        # Barra de confianza
        self._update_bar(0.0 if is_uncertain else confidence)

    def update_history(self, entries: list[tuple]):
        """
        Refresca la lista del historial.

        Args:
            entries: lista de tuplas (sena, timestamp) en orden cronológico
                     inverso (la más reciente primero).
                     Ejemplo: [("hola", "2026-03-20 10:01:05"), ...]
        """
        # Limpiar items anteriores
        for item in self._history_items:
            item.destroy()
        self._history_items.clear()

        if not entries:
            placeholder = tk.Label(
                self._history_frame,
                text="Sin traducciones aún",
                bg=_BG_PANEL, fg=_FG_MUTED,
                font=("Arial", 11),
                pady=16,
            )
            placeholder.pack()
            self._history_items.append(placeholder)
            return

        for sena, timestamp in entries[:self.MAX_HISTORY_ITEMS]:
            item = self._make_history_item(sena, timestamp)
            item.pack(fill="x", pady=(0, 2))
            self._history_items.append(item)

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------

    def _build(self):
        """Construye todos los widgets del panel."""
        self.configure(padx=16, pady=16)

        # ── Sección superior: seña actual ──────────────────────────────
        top = tk.Frame(self, bg=_BG)
        top.pack(fill="x", pady=(0, 12))

        tk.Label(
            top, text="SEÑA DETECTADA",
            bg=_BG, fg=_FG_MUTED,
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill="x")

        self._sign_label = tk.Label(
            top, text="—",
            bg=_BG, fg=_UNCERTAIN,
            font=("Arial", 52, "bold"),
            anchor="w",
            pady=4,
        )
        self._sign_label.pack(fill="x")

        self._conf_text = tk.Label(
            top, text="esperando seña...",
            bg=_BG, fg=_FG_MUTED,
            font=("Arial", 11),
            anchor="w",
        )
        self._conf_text.pack(fill="x")

        # ── Barra de confianza ─────────────────────────────────────────
        bar_container = tk.Frame(self, bg=_BG, pady=8)
        bar_container.pack(fill="x")

        self._bar_canvas = tk.Canvas(
            bar_container,
            height=8, bg=_BAR_EMPTY,
            highlightthickness=0,
        )
        self._bar_canvas.pack(fill="x")
        self._bar_fill = None   # rectángulo coloreado, creado al primer update

        # ── Separador ─────────────────────────────────────────────────
        tk.Frame(self, bg=_BORDER, height=1).pack(fill="x", pady=(4, 12))

        # ── Historial ─────────────────────────────────────────────────
        header = tk.Frame(self, bg=_BG)
        header.pack(fill="x", pady=(0, 8))

        tk.Label(
            header, text="HISTORIAL",
            bg=_BG, fg=_FG_MUTED,
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(side="left")

        self._clear_btn = tk.Button(
            header,
            text="Limpiar",
            bg=_BG, fg=_ACCENT,
            font=("Arial", 10),
            bd=0, relief="flat",
            cursor="hand2",
            activebackground=_BG,
            activeforeground=_FG_PRIMARY,
            command=self._handle_clear,
        )
        self._clear_btn.pack(side="right")

        # Frame scrollable para el historial
        scroll_container = tk.Frame(self, bg=_BG_PANEL, bd=0)
        scroll_container.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(scroll_container, orient="vertical", width=6)
        scrollbar.pack(side="right", fill="y")

        self._history_canvas = tk.Canvas(
            scroll_container,
            bg=_BG_PANEL,
            highlightthickness=0,
            yscrollcommand=scrollbar.set,
        )
        self._history_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self._history_canvas.yview)

        self._history_frame = tk.Frame(self._history_canvas, bg=_BG_PANEL)
        self._history_frame_id = self._history_canvas.create_window(
            (0, 0), window=self._history_frame, anchor="nw"
        )

        # Ajustar scroll region cuando cambia el contenido
        self._history_frame.bind("<Configure>", self._on_frame_configure)
        self._history_canvas.bind("<Configure>", self._on_canvas_configure)

        # Placeholder inicial
        self.update_history([])

    def _make_history_item(self, sena: str, timestamp: str) -> tk.Frame:
        """Crea un ítem individual del historial."""
        item = tk.Frame(
            self._history_frame,
            bg=_BG_ITEM,
            padx=10, pady=6,
        )

        # Nombre de la seña
        tk.Label(
            item,
            text=sena.replace("_", " ").capitalize(),
            bg=_BG_ITEM, fg=_FG_PRIMARY,
            font=("Arial", 12, "bold"),
            anchor="w",
        ).pack(side="left")

        # Timestamp (solo hora)
        time_str = timestamp.split(" ")[-1] if " " in timestamp else timestamp
        tk.Label(
            item,
            text=time_str,
            bg=_BG_ITEM, fg=_FG_MUTED,
            font=("Arial", 10),
            anchor="e",
        ).pack(side="right")

        return item

    # ------------------------------------------------------------------
    # Barra de confianza
    # ------------------------------------------------------------------

    def _update_bar(self, confidence: float):
        """Actualiza el ancho y color de la barra de confianza."""
        self._bar_canvas.update_idletasks()
        total_w = self._bar_canvas.winfo_width()
        if total_w <= 1:
            # El widget aún no tiene tamaño real — esperar al próximo ciclo
            self.after(50, lambda: self._update_bar(confidence))
            return

        fill_w = max(0, int(total_w * confidence))

        # Color según nivel de confianza
        if confidence >= 0.75:
            color = _BAR_HIGH
        elif confidence >= 0.5:
            color = _BAR_MID
        else:
            color = _BAR_LOW

        self._bar_canvas.delete("bar")
        if fill_w > 0:
            self._bar_canvas.create_rectangle(
                0, 0, fill_w, 8,
                fill=color, outline="",
                tags="bar",
            )

    # ------------------------------------------------------------------
    # Scroll y eventos
    # ------------------------------------------------------------------

    def _on_frame_configure(self, _event=None):
        self._history_canvas.configure(
            scrollregion=self._history_canvas.bbox("all")
        )

    def _on_canvas_configure(self, event):
        self._history_canvas.itemconfig(
            self._history_frame_id, width=event.width
        )

    def _handle_clear(self):
        """Limpia la vista y llama al callback externo si está definido."""
        self.update_history([])
        self._sign_label.configure(text="—", fg=_UNCERTAIN)
        self._conf_text.configure(text="esperando seña...", fg=_FG_MUTED)
        self._update_bar(0.0)
        if callable(self._on_clear):
            self._on_clear()