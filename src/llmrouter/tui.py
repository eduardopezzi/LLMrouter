"""Interactive Textual interface for LLMrouter operations."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from llmrouter.cli_panel import (
    _model_blocks,
    model_priorities,
    render_current_settings,
    render_panel_summary,
    render_usage_report,
    set_model_enabled,
    set_model_priority_order,
    set_provider_cost_order,
)
from llmrouter.config import Settings, reload_settings
from llmrouter.core.registry import ModelRegistry, load_model_registry


class HelpScreen(ModalScreen[None]):
    """Modal help screen for the keyboard-driven interface."""

    def compose(self) -> ComposeResult:
        yield Static(
            "[b]LLMrouter TUI — atalhos[/b]\n\n"
            "1/2/3/4  trocar abas\n"
            "↑/↓      selecionar modelo\n"
            "q / k    subir modelo na prioridade\n"
            "a / j    descer modelo na prioridade\n"
            "e        alternar enabled/disabled\n"
            "s        salvar alterações\n"
            "r        recarregar do disco\n"
            "x        restaurar ordem original do YAML\n"
            "Ctrl+Q   sair\n"
            "Esc      fechar esta ajuda\n\n"
            "As alterações só entram no catálogo depois de pressionar [b]s[/b].",
            id="help-dialog",
        )

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key in {"escape", "q"}:
            self.dismiss(None)


class LLMrouterTUI(App[None]):
    """Full-screen operator console for routing and model catalog management."""

    TITLE = "LLMrouter"
    SUB_TITLE = "Routing control center"

    CSS = """
    Screen {
        background: $surface;
    }
    TabbedContent {
        height: 1fr;
    }
    TabPane {
        padding: 1 2;
    }
    #overview-content, #routing-content, #usage-content {
        border: round $accent;
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }
    #models-help, #models-status, #routing-status {
        height: auto;
        padding: 0 1;
    }
    #models-table {
        height: 1fr;
        margin-top: 1;
    }
    #routing-controls {
        height: auto;
        margin: 1 0;
    }
    #provider-order {
        width: 1fr;
    }
    Button {
        margin-left: 1;
    }
    #help-dialog {
        width: 62;
        height: auto;
        padding: 2 3;
        border: thick $accent;
        background: $panel;
    }
    """

    BINDINGS = [
        Binding("1", "switch_tab('overview')", "Overview", priority=True),
        Binding("2", "switch_tab('routing')", "Routing", priority=True),
        Binding("3", "switch_tab('models')", "Models", priority=True),
        Binding("4", "switch_tab('usage')", "Usage", priority=True),
        Binding("q", "move_up", "Earlier", priority=True),
        Binding("k", "move_up", "Earlier", show=False, priority=True),
        Binding("a", "move_down", "Later", priority=True),
        Binding("j", "move_down", "Later", show=False, priority=True),
        Binding("e", "toggle_enabled", "Enable/disable", priority=True),
        Binding("s", "save", "Save", priority=True),
        Binding("x", "reset_order", "Reset order", priority=True),
        Binding("r", "refresh", "Reload", priority=True),
        Binding("question", "help", "Help", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        settings: Settings,
        *,
        models_file: str | Path | None = None,
        env_file: str | Path = ".env",
    ) -> None:
        super().__init__()
        self.settings = settings
        self.models_file = Path(models_file or settings.models_file)
        self.env_file = Path(env_file)
        self.catalog = ModelRegistry()
        self.active_catalog = ModelRegistry()
        self._order: list[str] = []
        self._enabled: dict[str, bool] = {}
        self._original_enabled: dict[str, bool] = {}
        self._dirty = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="models"):
            with TabPane("Overview", id="overview"):
                yield Static(id="overview-content")
            with TabPane("Routing", id="routing"):
                yield Static(id="routing-content")
                with Horizontal(id="routing-controls"):
                    yield Input(placeholder="zai,ollama,deepseek", id="provider-order")
                    yield Button("Save provider order", id="save-provider-order")
                yield Static(id="routing-status")
            with TabPane("Models", id="models"):
                yield Label(
                    "↑/↓ seleciona | q/k sobe | a/j desce | e alterna status | "
                    "s salva | r recarrega | x restaura ordem",
                    id="models-help",
                )
                yield Static(id="models-status")
                with VerticalScroll():
                    yield DataTable(id="models-table")
            with TabPane("Usage", id="usage"):
                yield Static(id="usage-content")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models-table", DataTable)
        table.cursor_type = "row"
        self._load_catalog()
        self._refresh_all()

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_refresh(self) -> None:
        self._load_catalog()
        self._refresh_all()
        self.notify("Catálogo recarregado")

    def _load_catalog(self) -> None:
        self.settings = reload_settings()
        self.models_file = Path(self.models_file or self.settings.models_file)
        self.catalog = load_model_registry(
            self.models_file,
            benchmark_catalog_path=self.settings.benchmarks.catalog_path,
            include_disabled=True,
        )
        self.active_catalog = load_model_registry(
            self.models_file,
            benchmark_catalog_path=self.settings.benchmarks.catalog_path,
        )
        rows = model_priorities(self.catalog, limit=None)
        self._order = [row.name for row in rows]
        self._enabled = {row.name: row.enabled for row in rows}
        self._original_enabled = dict(self._enabled)
        self._dirty = False

    def _refresh_all(self) -> None:
        self._update_overview()
        self._update_routing()
        self._update_models_table()
        self._update_usage()

    def _update_overview(self) -> None:
        self.query_one("#overview-content", Static).update(
            render_panel_summary(self.settings, self.active_catalog)
            + "\n\n"
            + "Telas: Overview | Routing | Models | Usage\n"
            + "Pressione ? para ajuda."
        )

    def _update_routing(self) -> None:
        order = ", ".join(self.settings.routing.provider_cost_order)
        self.query_one("#routing-content", Static).update(
            render_current_settings(self.settings, self.active_catalog)
        )
        self.query_one("#provider-order", Input).value = order

    def _update_models_table(self, cursor_row: int | None = None) -> None:
        table = self.query_one("#models-table", DataTable)
        table.clear(columns=True)
        table.add_column("#", width=5)
        table.add_column("Provider", width=10)
        table.add_column("Model", width=42)
        table.add_column("Tier", width=7)
        table.add_column("Status", width=10)
        table.add_column("Rollout", width=9)
        table.add_column("Roles", width=38)

        models = {model.name: model for model in self.catalog.all()}
        for index, name in enumerate(self._order, 1):
            model = models[name]
            status = "enabled" if self._enabled[name] else "disabled"
            table.add_row(
                str(index),
                model.provider.value,
                name,
                f"T{model.tier.value}",
                status,
                f"{model.rollout_percentage:g}%",
                ", ".join(sorted(model.capabilities)) or "-",
                key=name,
            )
        if cursor_row is not None and self._order:
            table.move_cursor(row=max(0, min(cursor_row, len(self._order) - 1)))
        marker = "modified" if self._dirty else "clean"
        self.query_one("#models-status", Static).update(
            f"{len(self._order)} modelos no catálogo | state={marker} | file={self.models_file}"
        )

    def _selected_index(self) -> int | None:
        if not self._order:
            return None
        row = self.query_one("#models-table", DataTable).cursor_row
        return max(0, min(row or 0, len(self._order) - 1))

    def action_move_up(self) -> None:
        index = self._selected_index()
        if index is None or index == 0:
            return
        self._order[index - 1], self._order[index] = self._order[index], self._order[index - 1]
        self._dirty = True
        self._update_models_table(index - 1)

    def action_move_down(self) -> None:
        index = self._selected_index()
        if index is None or index >= len(self._order) - 1:
            return
        self._order[index + 1], self._order[index] = self._order[index], self._order[index + 1]
        self._dirty = True
        self._update_models_table(index + 1)

    def action_toggle_enabled(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        name = self._order[index]
        self._enabled[name] = not self._enabled[name]
        self._dirty = True
        self._update_models_table(index)

    def action_save(self) -> None:
        if not self._order:
            return
        try:
            set_model_priority_order(self.models_file, self._order, include_disabled=True)
            for name, enabled in self._enabled.items():
                if enabled != self._original_enabled[name]:
                    set_model_enabled(self.models_file, name, enabled)
        except Exception as exc:
            self.notify(f"Falha ao salvar: {exc}", severity="error")
            return
        self._load_catalog()
        self._refresh_all()
        self.notify("Catálogo salvo")

    def action_reset_order(self) -> None:
        blocks = _model_blocks(self.models_file, include_disabled=True)
        self._order = [block.name for block in blocks]
        self._dirty = True
        self._update_models_table(0)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "save-provider-order":
            return
        value = self.query_one("#provider-order", Input).value
        providers = [item.strip() for item in value.split(",") if item.strip()]
        try:
            set_provider_cost_order(self.env_file, providers)
            self.settings = reload_settings()
            self._update_routing()
            self._update_overview()
            self.query_one("#routing-status", Static).update(
                f"Ordem salva: {', '.join(self.settings.routing.provider_cost_order)}"
            )
        except Exception as exc:
            self.query_one("#routing-status", Static).update(f"Erro: {exc}")

    def _update_usage(self) -> None:
        self.query_one("#usage-content", Static).update(
            render_usage_report(self.settings.evaluator.db_path, hours=6)
        )


def run_tui(
    settings: Settings,
    *,
    models_file: str | Path | None = None,
    env_file: str | Path = ".env",
) -> int:
    """Run the Textual LLMrouter operator console."""
    LLMrouterTUI(settings, models_file=models_file, env_file=env_file).run(mouse=True)
    return 0
