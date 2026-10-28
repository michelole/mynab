from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mynab.charts import (
    create_unified_plot,
    render_average_metrics_grid,
    render_in_grid,
    render_page_header,
)
from mynab.models import (
    CategoryInfo,
    DateLike,
)
from mynab.transforms import (
    _group_monthly_amounts,
    calculate_global_y_range,
    calculate_series_window_averages,
    filter_budget_by_selection,
    filter_transactions_by_selection,
    format_currency,
    get_full_month_period_index,
    get_global_month_range,
)


def create_category_plot(
    category_name: str,
    transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    global_month_range: pd.DatetimeIndex,
    categories_data: Sequence[CategoryInfo] | None,
    y_range: Sequence[float] | None = None,
) -> go.Figure | None:
    """Create comprehensive plot for a single category with enhanced metrics"""
    return create_unified_plot(
        category_name,
        transactions_df,
        budget_df,
        global_month_range,
        categories_data,
        "category",
        y_range,
    )


def calculate_summary_metrics(
    categories_data: Sequence[CategoryInfo],
    budget_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    selected_category_groups: Sequence[str],
    selected_categories: Sequence[str],
) -> tuple[float, float, dict[int, float]]:
    """Calculate summary metrics for all selected categories from full transaction history"""
    total_target_budget = float(
        sum(amt for cat in categories_data if (amt := cat.get("target_amount")) is not None)
    )
    total_available_budget = float(budget_df["available"].sum()) if not budget_df.empty else 0.0

    tx = filter_transactions_by_selection(
        transactions_df, selected_category_groups, selected_categories
    )
    monthly_totals: pd.Series | None = None
    if not tx.empty:
        monthly_totals = _group_monthly_amounts(tx)
        full_months = get_full_month_period_index(transactions_df)
        if full_months is not None and len(full_months) > 0:
            monthly_totals = monthly_totals.reindex(full_months, fill_value=0)

    averages = calculate_series_window_averages(monthly_totals)
    return total_target_budget, total_available_budget, averages


def _render_summary_metrics(
    total_target: float,
    total_available: float,
    averages: Mapping[int, float],
) -> None:
    """Display summary metrics at the top of the categories page"""
    st.header("📊 Summary Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="🎯 Total Target Budget",
            value=format_currency(total_target),
            help="Sum of all category target budgets",
        )
    with col2:
        st.metric(
            label="💰 Total Available Budget",
            value=format_currency(total_available),
            help="Sum of all category available budgets",
        )

    render_average_metrics_grid(averages, target_amount=total_target, short=False)


def _render_single_category_card(
    category_name: str,
    filtered_transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    global_month_range: pd.DatetimeIndex,
    categories_data: Sequence[CategoryInfo],
    category_y_range: Sequence[float] | None,
) -> None:
    """Render subheader and plot for a single category."""
    st.subheader(category_name)
    category_fig = create_category_plot(
        category_name,
        filtered_transactions_df,
        budget_df,
        global_month_range,
        categories_data,
        category_y_range,
    )
    if category_fig:
        st.plotly_chart(category_fig, width="stretch")
    else:
        st.info(f"No data available for {category_name}")


def _render_category_groups(
    filtered_categories_data: Sequence[CategoryInfo],
    filtered_transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    global_month_range: pd.DatetimeIndex,
    categories_data: Sequence[CategoryInfo],
    category_y_range: Sequence[float] | None,
) -> None:
    """Group and display plots for each category group"""
    categories_by_group: dict[str, list[str]] = {}
    for cat in filtered_categories_data:
        categories_by_group.setdefault(cat["group"], []).append(cat["name"])

    for group_name, category_names in categories_by_group.items():
        st.markdown("---")
        st.header(f"📂 {group_name}")
        render_in_grid(
            category_names,
            lambda cat_name: _render_single_category_card(
                cat_name,
                filtered_transactions_df,
                budget_df,
                global_month_range,
                categories_data,
                category_y_range,
            ),
        )


def _filter_categories_page_data(
    transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    categories_data: Sequence[CategoryInfo],
    start_date: DateLike | None,
    end_date: DateLike | None,
    selected_category_groups: Sequence[str],
    selected_categories: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[CategoryInfo]]:
    """Filter transactions, budget, and categories data for the categories page"""
    filtered_transactions_df = filter_transactions_by_selection(
        transactions_df,
        selected_category_groups,
        selected_categories,
        start_date=start_date,
        end_date=end_date,
    )
    filtered_budget_df = filter_budget_by_selection(
        budget_df, selected_category_groups, selected_categories
    )
    filtered_categories_data = [
        cat
        for cat in categories_data
        if cat["group"] in selected_category_groups and cat["name"] in selected_categories
    ]
    return filtered_transactions_df, filtered_budget_df, filtered_categories_data


def main() -> None:
    render_page_header("📋 Individual Category Analysis")

    if not st.session_state.get("data_loaded", False):
        st.error("Data not loaded. Please go to the main page first.")
        return

    transactions_df = st.session_state.transactions_df
    budget_df = st.session_state.budget_df
    categories_data = st.session_state.categories_data
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    selected_category_groups = st.session_state.selected_category_groups

    if transactions_df is None or budget_df is None or categories_data is None:
        st.error("Data not available. Please go to the main page first.")
        return

    raw_categories = st.session_state.get("selected_categories")
    selected_categories = (
        [str(c) for c in raw_categories] if isinstance(raw_categories, list) else []
    )

    filtered_transactions_df, budget_df, filtered_categories_data = _filter_categories_page_data(
        transactions_df,
        budget_df,
        categories_data,
        start_date,
        end_date,
        selected_category_groups,
        selected_categories,
    )
    filtered_category_names = [cat["name"] for cat in filtered_categories_data]

    global_month_range, _, _ = get_global_month_range(
        filtered_transactions_df, start_date, end_date
    )

    total_target, total_available, averages = calculate_summary_metrics(
        filtered_categories_data,
        budget_df,
        transactions_df,
        selected_category_groups,
        selected_categories,
    )

    _render_summary_metrics(total_target, total_available, averages)

    category_y_range: list[float] | None = None
    if st.session_state.get("global_scale_enabled", False) and filtered_category_names:
        category_y_range = calculate_global_y_range(
            filtered_transactions_df,
            budget_df,
            categories_data,
            "category",
            filtered_category_names,
        )

    _render_category_groups(
        filtered_categories_data,
        filtered_transactions_df,
        budget_df,
        global_month_range,
        categories_data,
        category_y_range,
    )


if __name__ == "__main__":
    main()
