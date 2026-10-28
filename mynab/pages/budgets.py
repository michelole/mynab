from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mynab.api import get_category_balances
from mynab.app import _get_api_token, ensure_data_loaded
from mynab.charts import (
    add_bar_trace,
    add_forecast_traces,
    add_moving_average_trace,
    apply_standard_plot_layout,
    render_average_metrics_grid,
    render_in_grid,
    render_page_header,
)
from mynab.models import AVERAGE_WINDOWS, CategoryInfo
from mynab.transforms import (
    format_currency,
    get_global_month_range,
    get_moving_average_window,
    map_account_net_worth_to_months,
    moving_average_for_plot_months,
)


def _selected_categories() -> list[str]:
    """Category names currently selected in the sidebar."""
    raw = st.session_state.get("selected_categories")
    return [str(c) for c in raw] if isinstance(raw, list) else []


def _category_series(balances_df: pd.DataFrame, category_name: str) -> pd.Series:
    """Full-history end-of-month balance of one category, indexed by month period."""
    sub = balances_df[balances_df["category"] == category_name].sort_values("month")
    return pd.Series(
        sub["balance"].to_numpy(dtype=float),
        index=pd.PeriodIndex(sub["month"], freq="M"),
    )


def _category_target(categories_data: Sequence[CategoryInfo], category_name: str) -> float | None:
    """Target budget amount of a category, if any."""
    return next(
        (
            float(amt)
            for cat in categories_data
            if cat["name"] == category_name and (amt := cat.get("target_amount")) is not None
        ),
        None,
    )


def _create_budget_plot(
    full_series: pd.Series,
    months: pd.DatetimeIndex,
    y_range: Sequence[float] | None = None,
) -> go.Figure:
    """Bar chart of end-of-month balance with moving average and trendline."""
    months_to_plot = [str(m) for m in months.strftime("%Y-%m")]
    values = map_account_net_worth_to_months(full_series, months_to_plot)
    plot_df = pd.DataFrame(
        {"month_date": months, "month": months_to_plot, "balance": values}
    ).sort_values("month_date")

    fig = go.Figure()
    add_bar_trace(fig, plot_df["month"], plot_df["balance"], "Left at end of month", "#2ca02c")

    ma_window = get_moving_average_window()
    ma_plot = moving_average_for_plot_months(full_series, months_to_plot, window=ma_window)
    add_moving_average_trace(fig, months_to_plot, ma_plot, ma_window=ma_window)

    add_forecast_traces(
        fig,
        plot_df["month"],
        plot_df["month_date"],
        plot_df["balance"],
        abs_forecast=False,
    )
    return apply_standard_plot_layout(
        fig, height=400, y_range=y_range, yaxis_title="Left at end of month (€)"
    )


def _calculate_global_y_range(
    balances_df: pd.DataFrame, category_names: Sequence[str], months: pd.DatetimeIndex
) -> list[float] | None:
    """Shared y-axis range covering the plotted balances of all given categories."""
    values = balances_df[
        balances_df["category"].isin(category_names) & balances_df["month"].isin(months)
    ]["balance"]
    if values.empty:
        return None
    min_val, max_val = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    padding = (max_val - min_val) * 0.1
    return [min_val - padding if min_val < 0 else 0.0, max_val + padding]


def _render_budget_card(
    category_name: str,
    balances_df: pd.DataFrame,
    months: pd.DatetimeIndex,
    categories_data: Sequence[CategoryInfo],
    y_range: Sequence[float] | None = None,
) -> None:
    """Render metrics and chart of the money left at the end of each month for a category."""
    st.subheader(category_name)
    full_series = _category_series(balances_df, category_name)
    if full_series.empty:
        st.info(f"No data available for {category_name}")
        return

    target_amount = _category_target(categories_data, category_name)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "🎯 Target Budget",
            format_currency(target_amount)
            if target_amount is not None and target_amount > 0
            else "No target set",
        )
    with col2:
        st.metric("💰 Available Budget", format_currency(float(full_series.iloc[-1])))
    render_average_metrics_grid(
        {window: float(full_series.tail(window).min()) for window in AVERAGE_WINDOWS},
        short=False,
        stat="Minimum",
    )

    st.plotly_chart(_create_budget_plot(full_series, months, y_range), width="stretch")


def _render_groups(
    balances_df: pd.DataFrame,
    category_names: Sequence[str],
    selected_groups: Sequence[str],
    months: pd.DatetimeIndex,
    categories_data: Sequence[CategoryInfo],
) -> None:
    """Render category cards grouped by category group."""
    available = set(balances_df["category"])
    categories_by_group: dict[str, list[str]] = {}
    for cat in categories_data:
        if (
            cat["group"] in selected_groups
            and cat["name"] in category_names
            and cat["name"] in available
        ):
            categories_by_group.setdefault(cat["group"], []).append(cat["name"])

    y_range: list[float] | None = None
    if st.session_state.get("global_scale_enabled", False):
        y_range = _calculate_global_y_range(
            balances_df, [n for names in categories_by_group.values() for n in names], months
        )

    for group_name, group_categories in categories_by_group.items():
        st.markdown("---")
        st.header(f"📂 {group_name}")
        render_in_grid(
            group_categories,
            lambda name: _render_budget_card(name, balances_df, months, categories_data, y_range),
        )


def main() -> None:
    render_page_header("🪣 Budgets")

    if not ensure_data_loaded():
        return

    token = _get_api_token()
    categories_data = st.session_state.categories_data
    if not token or not categories_data or not st.session_state.get("budget_id"):
        st.error("Data not available. Please go to the main page first.")
        return

    with st.spinner("Fetching monthly category balances..."):
        balances_df = get_category_balances(token, st.session_state.budget_id, categories_data)
    if balances_df.empty:
        st.info("No monthly category balances available.")
        return

    months, _, _ = get_global_month_range(
        None, st.session_state.start_date, st.session_state.end_date
    )
    _render_groups(
        balances_df,
        _selected_categories(),
        st.session_state.selected_category_groups,
        months,
        categories_data,
    )


if __name__ == "__main__":
    main()
