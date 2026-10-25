from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mynab.utils import (
    CategoryInfo,
    add_bar_trace,
    add_forecast_traces,
    add_moving_average_trace,
    add_target_goal_trace,
    apply_standard_plot_layout,
    build_complete_monthly_df,
    build_overview_monthly_series,
    calculate_global_y_range,
    compute_overview_monthly_totals,
    create_unified_plot,
    filter_budget_by_selection,
    filter_categorized_expense_transactions,
    filter_income_transactions,
    filter_transactions_by_selection,
    format_currency,
    get_global_month_range,
    get_moving_average_window,
    moving_average_for_plot_months,
    render_in_grid,
    render_page_header,
    resolve_selected_category_groups,
)

OVERVIEW_COLORS: dict[str, str] = {
    "total_income": "#2ca02c",
    "total_expense": "#ff7f0e",
    "total_net_income": "#1f77b4",
}


def create_category_group_plot(
    group_name: str,
    transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    month_range: pd.DatetimeIndex,
    categories_data: Sequence[CategoryInfo] | None,
    y_range: Sequence[float] | None = None,
) -> go.Figure | None:
    """Create comprehensive plot for a single category group"""
    return create_unified_plot(
        group_name,
        transactions_df,
        budget_df,
        month_range,
        categories_data,
        "category_group",
        y_range,
    )


def _filter_and_aggregate_overview_data(
    data_type: str, transactions_df: pd.DataFrame
) -> tuple[pd.DataFrame | None, str]:
    """Filter transactions by overview data type and aggregate by month"""
    color = OVERVIEW_COLORS.get(data_type, "#1f77b4")
    monthly_series = compute_overview_monthly_totals(
        transactions_df, data_type, abs_expense=False
    )
    if monthly_series.empty:
        return None, color

    monthly_data = monthly_series.reset_index()
    monthly_data.columns = ["month", "amount"]
    monthly_data["month"] = monthly_data["month"].astype(str)
    return monthly_data, color


def _add_overview_forecast_traces(
    fig: go.Figure, complete_monthly_data: pd.DataFrame, data_type: str
) -> pd.DatetimeIndex | None:
    """Add forecast trend line and 3-month extension to overview figure"""
    if len(complete_monthly_data) < 3:
        return None

    actual_data = complete_monthly_data[complete_monthly_data["amount"] != 0].copy()
    if len(actual_data) < 3:
        return None

    is_expense = data_type == "total_expense"
    values = actual_data["amount"].abs() if is_expense else actual_data["amount"]
    return add_forecast_traces(
        fig,
        actual_data["month"],
        actual_data["month_date"],
        values,
        abs_forecast=is_expense,
    )


def create_comprehensive_plot(
    data_type: str,
    transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    global_month_range: pd.DatetimeIndex,
    y_range: Sequence[float] | None = None,
    target_goal: float | None = None,
) -> go.Figure | None:
    """Create comprehensive plot for total income, total expense, or total net income"""
    monthly_data, color = _filter_and_aggregate_overview_data(
        data_type, transactions_df
    )
    if monthly_data is None:
        return None

    complete_monthly_data = build_complete_monthly_df(global_month_range, monthly_data)
    months_to_plot: list[str] = [
        str(m) for m in complete_monthly_data["month"].tolist()
    ]
    y_values = (
        complete_monthly_data["amount"].abs()
        if data_type == "total_expense"
        else complete_monthly_data["amount"]
    )
    title_label = data_type.replace("_", " ").title()

    fig = go.Figure()
    add_bar_trace(
        fig,
        months_to_plot,
        y_values,
        f"Actual {title_label}",
        color,
        hover_label=title_label,
    )

    ma_window = get_moving_average_window()
    full_monthly = build_overview_monthly_series(data_type)
    ma_plot = moving_average_for_plot_months(
        full_monthly, months_to_plot, window=ma_window
    )
    add_moving_average_trace(fig, months_to_plot, ma_plot, ma_window=ma_window)

    future_months = _add_overview_forecast_traces(fig, complete_monthly_data, data_type)
    if data_type == "total_expense":
        add_target_goal_trace(fig, months_to_plot, future_months, target_goal)

    return apply_standard_plot_layout(fig, height=500, y_range=y_range)


def _mean_or_zero(series: pd.Series) -> float:
    """Return the mean of a Series, or 0 if empty."""
    return float(series.mean()) if not series.empty else 0.0


def _render_overview_summary_metrics(
    filtered_transactions_df: pd.DataFrame,
) -> None:
    """Render total and monthly-average summary metrics for the overview page."""
    income_tx = filter_income_transactions(filtered_transactions_df)
    expense_tx = filter_categorized_expense_transactions(filtered_transactions_df)

    total_income = float(income_tx["amount"].sum()) if not income_tx.empty else 0.0
    total_expenses = float(expense_tx["amount"].sum()) if not expense_tx.empty else 0.0
    total_net_income = total_income + total_expenses

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Income", format_currency(total_income))
    with col2:
        st.metric("Total Expenses", format_currency(total_expenses))
    with col3:
        st.metric("Total Net Income", format_currency(total_net_income))

    monthly_income = compute_overview_monthly_totals(
        filtered_transactions_df, "total_income"
    )
    monthly_expenses = compute_overview_monthly_totals(
        filtered_transactions_df, "total_expense", abs_expense=False
    )
    monthly_net = compute_overview_monthly_totals(
        filtered_transactions_df, "total_net_income"
    )

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Avg Monthly Income", format_currency(_mean_or_zero(monthly_income)))
    with col5:
        st.metric(
            "Avg Monthly Expenses", format_currency(_mean_or_zero(monthly_expenses))
        )
    with col6:
        st.metric("Avg Monthly Net Income", format_currency(_mean_or_zero(monthly_net)))


def _render_overview_charts_row(
    filtered_transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    global_month_range: pd.DatetimeIndex,
    overview_y_range: Sequence[float] | None,
    categories_data: Sequence[CategoryInfo] | None,
    selected_category_groups: Sequence[str],
) -> None:
    """Render the 3 main overview charts (Total Income, Total Expenses, Net Income)."""
    st.header("📊 Overview Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("💰 Total Income")
        income_fig = create_comprehensive_plot(
            "total_income",
            filtered_transactions_df,
            budget_df,
            global_month_range,
            overview_y_range,
        )
        if income_fig:
            st.plotly_chart(income_fig, width="stretch")
        else:
            st.info("No income data available")

    with col2:
        st.subheader("💸 Total Expenses")
        total_target_goal = float(
            sum(
                amt
                for cat in (categories_data or [])
                if cat["group"] in selected_category_groups
                and (amt := cat.get("target_amount")) is not None
            )
        )
        expense_fig = create_comprehensive_plot(
            "total_expense",
            filtered_transactions_df,
            budget_df,
            global_month_range,
            overview_y_range,
            target_goal=total_target_goal,
        )
        if expense_fig:
            st.plotly_chart(expense_fig, width="stretch")
        else:
            st.info("No expense data available")

    with col3:
        st.subheader("📈 Monthly Net Income")
        net_income_fig = create_comprehensive_plot(
            "total_net_income",
            filtered_transactions_df,
            budget_df,
            global_month_range,
            overview_y_range,
        )
        if net_income_fig:
            st.plotly_chart(net_income_fig, width="stretch")
        else:
            st.info("No net income data available")


def _render_single_group_card(
    group_name: str,
    filtered_transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    global_month_range: pd.DatetimeIndex,
    categories_data: Sequence[CategoryInfo] | None,
    category_group_y_range: Sequence[float] | None,
) -> None:
    """Render subheader and plot for a single category group."""
    st.subheader(group_name)
    group_fig = create_category_group_plot(
        group_name,
        filtered_transactions_df,
        budget_df,
        global_month_range,
        categories_data,
        category_group_y_range,
    )
    if group_fig:
        st.plotly_chart(group_fig, width="stretch")
    else:
        st.info(f"No data available for {group_name}")


def _compute_overview_y_ranges(
    filtered_transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    categories_data: Sequence[CategoryInfo] | None,
    selected_category_groups: Sequence[str],
) -> tuple[list[float] | None, list[float] | None]:
    """Calculate global y-axis ranges for overview and category-group plots when enabled."""
    if not st.session_state.get("global_scale_enabled", False):
        return None, None

    overview_y_range = calculate_global_y_range(
        filtered_transactions_df, budget_df, categories_data, "overview"
    )
    category_group_y_range = (
        calculate_global_y_range(
            filtered_transactions_df,
            budget_df,
            categories_data,
            "category_group",
            selected_category_groups,
        )
        if selected_category_groups
        else None
    )
    return overview_y_range, category_group_y_range


def _validate_overview_session() -> bool:
    """Validate required session state for the overview page."""
    if not st.session_state.get("data_loaded", False):
        st.error("Data not loaded. Please go back to the main page.")
        return False
    if (
        st.session_state.get("start_date") is None
        or st.session_state.get("end_date") is None
    ):
        st.error(
            "Month range not set. Please configure the month range in the sidebar."
        )
        return False
    if st.session_state.get("transactions_df") is None:
        st.error("No transaction data available.")
        return False
    return True


def main() -> None:
    render_page_header("💰 YNAB Budget Dashboard")

    if not _validate_overview_session():
        return

    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    transactions_df = st.session_state.get("transactions_df")
    budget_df = st.session_state.get("budget_df")
    categories_data = st.session_state.get("categories_data")
    selected_category_groups = resolve_selected_category_groups()
    raw_categories = st.session_state.get("selected_categories")
    selected_categories = (
        [str(c) for c in raw_categories] if isinstance(raw_categories, list) else []
    )

    filtered_transactions_df = filter_transactions_by_selection(
        transactions_df,
        selected_category_groups,
        selected_categories,
        start_date=start_date,
        end_date=end_date,
        preserve_income=True,
    )
    budget_df = filter_budget_by_selection(
        budget_df, selected_category_groups, selected_categories
    )
    global_month_range, _, _ = get_global_month_range(
        filtered_transactions_df, start_date, end_date
    )

    overview_y_range, category_group_y_range = _compute_overview_y_ranges(
        filtered_transactions_df, budget_df, categories_data, selected_category_groups
    )

    _render_overview_summary_metrics(filtered_transactions_df)
    st.markdown("---")
    _render_overview_charts_row(
        filtered_transactions_df,
        budget_df,
        global_month_range,
        overview_y_range,
        categories_data,
        selected_category_groups,
    )
    st.markdown("---")

    st.header("📊 Category Group Analysis")
    render_in_grid(
        selected_category_groups,
        lambda group_name: _render_single_group_card(
            group_name,
            filtered_transactions_df,
            budget_df,
            global_month_range,
            categories_data,
            category_group_y_range,
        ),
    )


if __name__ == "__main__":
    main()
