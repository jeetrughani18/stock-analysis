"""
Stock Performance Analysis — Cross-Platform App
=================================================
Flet-based GUI that runs on Windows, macOS, Android & iOS.
All metric logic is imported from main.py.
"""

import os
import threading

import flet as ft
import pandas as pd

from main import (
    BENCHMARK_MAP,
    download_data,
    align_and_compute_returns,
    compute_all_metrics,
)


def main(page: ft.Page) -> None:
    # ── Page configuration ────────────────────
    page.title = "Stock Performance Analysis"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30
    page.scroll = ft.ScrollMode.AUTO

    # ── Colour palette ────────────────────────
    PRIMARY = ft.Colors.INDIGO
    ACCENT = ft.Colors.INDIGO_100

    # ── Input controls ────────────────────────
    stock_input = ft.TextField(
        label="Stock Symbol",
        hint_text="e.g. RELIANCE.NS",
        prefix_icon=ft.Icons.SHOW_CHART,
        border_radius=10,
        width=300,
    )

    bench_dropdown = ft.Dropdown(
        label="Benchmark Index",
        width=300,
        border_radius=10,
        options=[ft.dropdown.Option(key) for key in BENCHMARK_MAP],
        value="NIFTY50",
    )

    rf_input = ft.TextField(
        label="Risk-Free Rate (annual)",
        hint_text="e.g. 0.07 for 7 %",
        prefix_icon=ft.Icons.PERCENT,
        border_radius=10,
        width=300,
        keyboard_type=ft.KeyboardType.NUMBER,
    )

    # ── Status / progress ─────────────────────
    progress_ring = ft.ProgressRing(visible=False, width=24, height=24)
    status_text = ft.Text(value="", size=14)

    # ── Results area ──────────────────────────
    results_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Metric", weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Value", weight=ft.FontWeight.BOLD)),
        ],
        rows=[],
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=10,
        heading_row_color=ACCENT,
        column_spacing=40,
    )

    results_container = ft.Container(
        content=results_table,
        visible=False,
        padding=10,
    )

    save_button = ft.ElevatedButton(
        content=ft.Text("Save to CSV"),
        icon=ft.Icons.SAVE_ALT,
        visible=False,
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.GREEN_600,
            color=ft.Colors.WHITE,
            shape=ft.RoundedRectangleBorder(radius=10),
        ),
    )

    # ── Shared state ──────────────────────────
    last_result: dict = {"df": None, "symbol": ""}

    # ── Analyse handler ───────────────────────
    def run_analysis(e: ft.ControlEvent) -> None:
        symbol = stock_input.value.strip().upper()
        bench_name = bench_dropdown.value
        rf_text = rf_input.value.strip()

        # Validation
        if not symbol:
            show_status("⚠️  Please enter a stock symbol.", ft.Colors.RED_700)
            return
        if not bench_name:
            show_status("⚠️  Please select a benchmark.", ft.Colors.RED_700)
            return
        try:
            risk_free = float(rf_text)
        except (ValueError, TypeError):
            show_status("⚠️  Risk-free rate must be a number.", ft.Colors.RED_700)
            return

        # Show progress
        progress_ring.visible = True
        results_container.visible = False
        save_button.visible = False
        show_status("⏳  Downloading data…", PRIMARY)
        page.update()

        # Run heavy work off the UI thread
        def _worker():
            try:
                bench_ticker = BENCHMARK_MAP[bench_name]
                stock_prices = download_data(symbol)
                bench_prices = download_data(bench_ticker)

                if stock_prices is None and bench_prices is None:
                    _ui(lambda: show_status(
                        f"❌  Could not download data for '{symbol}' and '{bench_name}'.",
                        ft.Colors.RED_700))
                    return
                if stock_prices is None:
                    _ui(lambda: show_status(
                        f"❌  Could not download data for stock '{symbol}'.",
                        ft.Colors.RED_700))
                    return
                if bench_prices is None:
                    _ui(lambda: show_status(
                        f"❌  Could not download data for benchmark '{bench_name}'.",
                        ft.Colors.RED_700))
                    return

                _ui(lambda: show_status("📐  Aligning dates & computing returns…", PRIMARY))

                df = align_and_compute_returns(stock_prices, bench_prices)
                if len(df) < 2:
                    _ui(lambda: show_status(
                        f"❌  Only {len(df)} aligned trading days — need at least 2.",
                        ft.Colors.RED_700))
                    return

                _ui(lambda: show_status("📊  Computing metrics…", PRIMARY))

                result = compute_all_metrics(
                    df, df["stock_price"], df["bench_price"],
                    risk_free, symbol, bench_name,
                )

                # Store for CSV export
                last_result["df"] = result
                last_result["symbol"] = symbol

                # Build table rows
                rows = []
                for _, row in result.iterrows():
                    rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text(str(row["Metric"]))),
                        ft.DataCell(ft.Text(str(row["Value"]))),
                    ]))

                def _show():
                    results_table.rows = rows
                    results_container.visible = True
                    save_button.visible = True
                    show_status(f"✅  Analysis complete for {symbol}", ft.Colors.GREEN_700)

                _ui(_show)

            except Exception as exc:
                _ui(lambda: show_status(f"❌  Error: {exc}", ft.Colors.RED_700))
            finally:
                _ui(lambda: _hide_progress())

        def _ui(fn):
            fn()
            page.update()

        def _hide_progress():
            progress_ring.visible = False
            page.update()

        threading.Thread(target=_worker, daemon=True).start()

    # ── Save handler ──────────────────────────
    def save_csv(e: ft.ControlEvent) -> None:
        df = last_result.get("df")
        symbol = last_result.get("symbol", "STOCK")
        if df is None:
            return
        filename = f"{symbol}_performance_metrics.csv"
        filepath = os.path.join(os.getcwd(), filename)
        df.to_csv(filepath, index=False)
        show_status(f"💾  Saved to {filepath}", ft.Colors.GREEN_700)
        page.update()

    save_button.on_click = save_csv

    # ── Helper ────────────────────────────────
    def show_status(msg: str, colour=None):
        status_text.value = msg
        status_text.color = colour
        page.update()

    # ── Analyse button ────────────────────────
    analyse_button = ft.ElevatedButton(
        content=ft.Text("Analyse"),
        icon=ft.Icons.ANALYTICS,
        on_click=run_analysis,
        width=200,
        height=48,
        style=ft.ButtonStyle(
            bgcolor=PRIMARY,
            color=ft.Colors.WHITE,
            shape=ft.RoundedRectangleBorder(radius=10),
        ),
    )

    # ── Layout ────────────────────────────────
    page.add(
        ft.Text(
            "📈 Stock Performance Analysis",
            size=26,
            weight=ft.FontWeight.BOLD,
            color=PRIMARY,
        ),
        ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
        # Input card
        ft.Card(
            content=ft.Container(
                padding=20,
                content=ft.Column(
                    [
                        ft.Text("Enter details", size=16, weight=ft.FontWeight.W_600),
                        ft.ResponsiveRow(
                            [
                                ft.Container(stock_input, col={"sm": 12, "md": 4}),
                                ft.Container(bench_dropdown, col={"sm": 12, "md": 4}),
                                ft.Container(rf_input, col={"sm": 12, "md": 4}),
                            ],
                        ),
                        ft.Row(
                            [analyse_button, progress_ring],
                            alignment=ft.MainAxisAlignment.START,
                            spacing=15,
                        ),
                    ],
                    spacing=15,
                ),
            ),
            elevation=3,
        ),
        ft.Container(height=10),
        status_text,
        ft.Container(height=10),
        # Results card
        ft.Card(
            content=results_container,
            elevation=3,
        ),
        ft.Container(height=10),
        save_button,
    )


ft.run(main)
