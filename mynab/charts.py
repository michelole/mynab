"""Plotly chart builders and shared Streamlit UI rendering helpers."""

from collections.abc import Callable, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mynab.config import settings
from mynab.models import (
    MIN_MONTHS_FOR_FORECAST,
    CategoryInfo,
    MovingAverageWindow,
)
from mynab.transforms import (
    _group_monthly_amounts,
    _resolve_unfiltered_transactions,
    build_complete_monthly_df,
    build_entity_monthly_series,
    calculate_category_available_budget,
    calculate_category_group_available_budget,
    calculate_entity_window_averages,
    calculate_forecast_trend,
    format_currency,
    format_target_delta,
    get_moving_average_window,
    moving_average_for_plot_months,
    moving_average_label,
)

DASHBOARD_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
</style>
"""


def inject_custom_css() -> None:
    """Inject shared dashboard CSS styles into the current Streamlit page."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def render_page_header(title: str) -> None:
    """Render shared CSS and styled main header for a page."""
    inject_custom_css()
    st.markdown(f'<h1 class="main-header">{title}</h1>', unsafe_allow_html=True)


def render_in_grid[T](
    items: Sequence[T],
    render_item_fn: Callable[[T], None],
    cols_per_row: int = 2,
) -> None:
    """Render a list of items across a multi-column Streamlit grid."""
    for i in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, items[i : i + cols_per_row], strict=False):
            with col:
                render_item_fn(item)


def add_bar_trace(
    fig: go.Figure,
    x_months: Sequence[str] | pd.Series,
    y_values: Sequence[float] | pd.Series,
    name: str,
    color: str,
    hover_label: str | None = None,
) -> None:
    """Add a formatted currency Bar trace to a Plotly figure."""
    label = hover_label or name
    fig.add_trace(
        go.Bar(
            x=x_months,
            y=y_values,
            name=name,
            marker_color=color,
            opacity=0.8,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{customdata}}<br><extra></extra>",
            customdata=[format_currency(float(val)) for val in y_values],
        )
    )


def add_moving_average_trace(
    fig: go.Figure,
    months_to_plot: Sequence[str],
    ma_plot: Sequence[float | None] | None,
    ma_window: MovingAverageWindow | None = None,
) -> None:
    """Add a formatted moving average Scatter trace to a Plotly figure."""
    if ma_plot is None:
        return
    resolved_window = ma_window if ma_window is not None else get_moving_average_window()
    ma_label = moving_average_label(resolved_window)
    fig.add_trace(
        go.Scatter(
            x=months_to_plot,
            y=ma_plot,
            name=ma_label,
            line={"color": "#1f77b4", "width": 2, "dash": "dot"},
            mode="lines",
            hovertemplate=f"<b>%{{x}}</b><br>{ma_label}: %{{customdata}}<br><extra></extra>",
            customdata=[format_currency(v) if v is not None else "€0" for v in ma_plot],
        )
    )


def add_forecast_traces(
    fig: go.Figure,
    months_series: Sequence[str] | pd.Series,
    month_dates_series: pd.Series,
    values_for_regression: pd.Series,
    *,
    abs_forecast: bool = True,
    trend_name: str = "12-Month Forecast Trend",
) -> pd.DatetimeIndex | None:
    """Add linear regression trend line and 3-month forecast extension traces."""
    if len(values_for_regression) < MIN_MONTHS_FOR_FORECAST:
        return None

    trend_line, forecast = calculate_forecast_trend(values_for_regression)
    fig.add_trace(
        go.Scatter(
            x=months_series,
            y=trend_line,
            name=trend_name,
            line={"color": "#d62728", "width": 2},
            mode="lines",
            hovertemplate=f"<b>%{{x}}</b><br>{trend_name}: %{{customdata}}<br><extra></extra>",
            customdata=[format_currency(float(val)) for val in trend_line],
        )
    )

    last_month_date = pd.Timestamp(month_dates_series.iloc[-1])
    future_months = pd.date_range(
        start=last_month_date + pd.DateOffset(months=1),
        periods=settings.forecast_periods,
        freq="MS",
    )
    forecast_values = forecast.abs() if abs_forecast else forecast
    fig.add_trace(
        go.Scatter(
            x=future_months.strftime("%Y-%m"),
            y=forecast_values,
            name="Forecast (Next 3 Months)",
            line={"color": "#d62728", "width": 2, "dash": "dash"},
            mode="lines",
            hovertemplate="<b>%{x}</b><br>Forecast: %{customdata}<br><extra></extra>",
            customdata=[format_currency(float(val)) for val in forecast_values],
        )
    )
    return future_months


def add_target_goal_trace(
    fig: go.Figure,
    base_months: Sequence[str],
    future_months: pd.DatetimeIndex | None,
    target_amount: float | None,
) -> None:
    """Add a horizontal target goal line across historical and forecast months."""
    if target_amount is None or target_amount <= 0:
        return
    all_months = list(base_months)
    if future_months is not None:
        all_months.extend(str(m) for m in future_months.strftime("%Y-%m"))
    formatted_target = format_currency(target_amount)
    fig.add_trace(
        go.Scatter(
            x=all_months,
            y=[target_amount] * len(all_months),
            name=f"Target Goal ({formatted_target})",
            line={"color": "#2ca02c", "width": 3, "dash": "solid"},
            mode="lines",
            hovertemplate="<b>%{x}</b><br>Target Goal: %{customdata}<br><extra></extra>",
            customdata=[formatted_target] * len(all_months),
        )
    )


def apply_standard_plot_layout(
    fig: go.Figure,
    height: int = 400,
    y_range: Sequence[float] | None = None,
    yaxis_title: str = "Amount (€)",
) -> go.Figure:
    """Apply consistent layout settings to a Plotly figure."""
    layout_update: dict[str, object] = {
        "height": height,
        "showlegend": False,
        "hovermode": "x unified",
        "xaxis_title": "Month",
        "yaxis_title": yaxis_title,
        "barmode": "overlay",
    }
    if y_range is not None:
        layout_update["yaxis"] = {"range": y_range}
    fig.update_layout(**layout_update)
    return fig


def render_average_metrics_grid(
    averages: Mapping[int, float],
    target_amount: float | None = None,
    *,
    short: bool = True,
    stat: str = "Average",
) -> None:
    """Render a 2x2 grid of 3, 6, 12, and 24-month metrics (average by default)."""
    unit = ("Avg" if stat == "Average" else stat[:3]) if short else stat
    for row_windows in ((3, 6), (12, 24)):
        cols = st.columns(2)
        for col, window in zip(cols, row_windows, strict=True):
            with col:
                val = averages.get(window, 0.0)
                delta_text, delta_color = format_target_delta(
                    val, target_amount, zero_text="On target"
                )
                st.metric(
                    label=f"📊 Last {window} Months {unit}",
                    value=format_currency(val),
                    delta=delta_text,
                    delta_color=delta_color,
                    help=f"{stat} monthly amount over the last {window} months",
                )


def _prepare_unified_plot_data(
    name: str,
    transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    categories_data: Sequence[CategoryInfo] | None,
    plot_type: str,
) -> tuple[pd.DataFrame, float | None, dict[int, float], float] | None:
    """Filter data and calculate summary metrics for unified plot."""
    cats = categories_data or []
    if plot_type == "category_group":
        filtered_transactions = pd.DataFrame(
            transactions_df[transactions_df["category_group"] == name]
        ).copy()
        target_amount: float | None = float(
            sum(
                amt
                for cat in cats
                if cat["group"] == name and (amt := cat.get("target_amount")) is not None
            )
        )
        available_budget = calculate_category_group_available_budget(name, budget_df)
    else:
        filtered_transactions = pd.DataFrame(
            transactions_df[transactions_df["category"] == name]
        ).copy()
        target_amount = next(
            (
                float(amt)
                for cat in cats
                if cat["name"] == name and (amt := cat.get("target_amount")) is not None
            ),
            None,
        )
        available_budget = calculate_category_available_budget(name, budget_df)

    if filtered_transactions.empty:
        return None

    averages = calculate_entity_window_averages(plot_type, name, transactions_df)
    return filtered_transactions, target_amount, averages, available_budget


def _render_unified_plot_metrics(
    target_amount: float | None,
    averages: Mapping[int, float],
    available_budget: float,
) -> None:
    """Render Streamlit metrics for a unified category or category group plot."""
    col1, col2 = st.columns(2)
    with col1:
        target_val = (
            format_currency(target_amount)
            if (target_amount is not None and target_amount > 0)
            else "No target set"
        )
        st.metric(label="🎯 Target Budget", value=target_val)
    with col2:
        st.metric(
            label="💰 Available Budget",
            value=format_currency(available_budget),
        )

    render_average_metrics_grid(averages, target_amount=target_amount, short=True)


def _add_unified_plot_traces(
    fig: go.Figure,
    complete_monthly_data: pd.DataFrame,
    plot_type: str,
    name: str,
    target_amount: float | None,
    transactions_df: pd.DataFrame | None = None,
) -> None:
    """Add bar, moving average, forecast, and target goal traces to unified plot."""
    months_to_plot: list[str] = [str(m) for m in complete_monthly_data["month"].tolist()]
    abs_amounts = complete_monthly_data["amount"].abs()

    add_bar_trace(fig, months_to_plot, abs_amounts, "Actual Expenses", "#ff7f0e")

    ma_window = get_moving_average_window()
    full_series_abs = build_entity_monthly_series(
        plot_type,
        name,
        transactions_df=_resolve_unfiltered_transactions(transactions_df),
        abs_values=True,
    )
    ma_plot = moving_average_for_plot_months(full_series_abs, months_to_plot, window=ma_window)
    add_moving_average_trace(fig, months_to_plot, ma_plot, ma_window=ma_window)

    future_months = add_forecast_traces(
        fig,
        complete_monthly_data["month"],
        complete_monthly_data["month_date"],
        abs_amounts,
        abs_forecast=True,
    )
    add_target_goal_trace(fig, months_to_plot, future_months, target_amount)


def create_unified_plot(
    name: str,
    transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    global_month_range: pd.DatetimeIndex,
    categories_data: Sequence[CategoryInfo] | None,
    plot_type: str = "category",
    y_range: Sequence[float] | None = None,
) -> go.Figure | None:
    """Create comprehensive plot for either a category group or individual category."""
    prepared = _prepare_unified_plot_data(
        name, transactions_df, budget_df, categories_data, plot_type
    )
    if prepared is None:
        return None

    filtered_transactions, target_amount, averages, available_budget = prepared

    _render_unified_plot_metrics(target_amount, averages, available_budget)

    monthly_series = _group_monthly_amounts(filtered_transactions)
    monthly_expenses = monthly_series.reset_index()
    monthly_expenses["month"] = monthly_expenses["month"].astype(str)

    fig = go.Figure()
    if not monthly_expenses.empty:
        complete_monthly_data = build_complete_monthly_df(global_month_range, monthly_expenses)
        _add_unified_plot_traces(
            fig,
            complete_monthly_data,
            plot_type,
            name,
            target_amount,
            transactions_df=transactions_df,
        )

    return apply_standard_plot_layout(fig, height=400, y_range=y_range)
