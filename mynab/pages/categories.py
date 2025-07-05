import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
from mynab.utils import (
    calculate_moving_average,
    calculate_forecast_trend,
    calculate_category_averages,
    calculate_category_available_budget,
    filter_data_by_date_range,
    safe_strftime,
    get_global_month_range,
    get_excluded_groups,
    get_default_categories,
    create_unified_plot,
    calculate_global_y_range,
)

# Page configuration
st.set_page_config(
    page_title="Individual Category Analysis - YNAB Dashboard",
    page_icon="📋",
    layout="wide",
)

# Custom CSS for better styling
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


def create_category_plot(
    category_name,
    transactions_df,
    budget_df,
    global_month_range,
    categories_data,
    y_range=None,
):
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


def main():
    st.markdown(
        '<h1 class="main-header">📋 Individual Category Analysis</h1>',
        unsafe_allow_html=True,
    )

    # Check if data is loaded in session state
    if not st.session_state.get("data_loaded", False):
        st.error("Data not loaded. Please go to the main page first.")
        return

    # Get data from session state
    transactions_df = st.session_state.transactions_df
    budget_df = st.session_state.budget_df
    categories_data = st.session_state.categories_data
    category_groups = st.session_state.category_groups
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    selected_category_groups = st.session_state.selected_category_groups

    if transactions_df is None or budget_df is None or categories_data is None:
        st.error("Data not available. Please go to the main page first.")
        return

    # Filter out transactions with excluded category groups
    excluded_groups = get_excluded_groups()
    filtered_transactions_df = transactions_df[
        ~transactions_df["category_group"].isin(excluded_groups)
    ].copy()

    # Apply date filtering
    filtered_transactions_df = filter_data_by_date_range(
        filtered_transactions_df, start_date, end_date
    )

    # Get selected categories from session state
    selected_categories = st.session_state.get("selected_categories", [])

    # Filter categories based on selected groups and selected categories
    filtered_categories_data = [
        cat
        for cat in categories_data
        if cat["group"] in selected_category_groups
        and cat["name"] in selected_categories
    ]
    filtered_category_names = [cat["name"] for cat in filtered_categories_data]

    # Filter transactions to only include selected category groups and categories
    if (
        isinstance(filtered_transactions_df, pd.DataFrame)
        and not filtered_transactions_df.empty
    ):
        filtered_transactions_df = filtered_transactions_df[
            (filtered_transactions_df["category_group"].isin(selected_category_groups))
            & (filtered_transactions_df["category"].isin(selected_categories))
        ].copy()

    # Filter budget data to only include selected category groups and categories
    if (
        budget_df is not None
        and isinstance(budget_df, pd.DataFrame)
        and not budget_df.empty
    ):
        budget_df = budget_df[
            (budget_df["category_group"].isin(selected_category_groups))
            & (budget_df["category"].isin(selected_categories))
        ].copy()

    # Calculate summary metrics
    def calculate_summary_metrics(categories_data, budget_df, filtered_transactions_df):
        """Calculate summary metrics for all categories"""
        # Total target budget
        total_target_budget = sum(
            cat.get("target_amount", 0) or 0
            for cat in categories_data
            if cat.get("target_amount") is not None
        )

        # Total available budget
        total_available_budget = (
            budget_df["available"].sum() if not budget_df.empty else 0
        )

        # Last 12 months average for all transactions
        if not filtered_transactions_df.empty:
            # Use all transactions (both income and expenses)
            all_transactions_df = filtered_transactions_df.copy()
            all_transactions_df["date"] = pd.to_datetime(all_transactions_df["date"])
            all_transactions_df["month"] = all_transactions_df["date"].dt.to_period("M")
            monthly_totals = all_transactions_df.groupby("month")["amount"].sum()
            if len(monthly_totals) >= 12:
                last_12_months_avg = abs(monthly_totals.tail(12).mean())
            else:
                last_12_months_avg = (
                    abs(monthly_totals.mean()) if not monthly_totals.empty else 0
                )
        else:
            last_12_months_avg = 0

        # Sum of suggested budgets
        total_suggested_budget = 0
        for cat in categories_data:
            category_name = cat["name"]
            avg_12_months = calculate_category_averages(
                category_name, filtered_transactions_df, 12
            )
            available_budget = calculate_category_available_budget(
                category_name, budget_df
            )
            suggested_budget = -(available_budget - (avg_12_months * 12)) / 12
            total_suggested_budget += suggested_budget

        return (
            total_target_budget,
            total_available_budget,
            last_12_months_avg,
            total_suggested_budget,
        )

    # Calculate summary metrics
    total_target, total_available, last_12m_avg, total_suggested = (
        calculate_summary_metrics(
            filtered_categories_data, budget_df, filtered_transactions_df
        )
    )

    # Display summary metrics at the top
    st.header("📊 Summary Metrics")

    # Create metrics in a 2x2 grid (matching individual category order)
    col1, col2 = st.columns(2)

    with col1:
        # Target Budget - no delta for total target
        st.metric(
            label="🎯 Total Target Budget",
            value=f"€{total_target:,.0f}",
            help="Sum of all category target budgets",
        )

        # Last 12 Months Average - delta as percentage vs target
        if total_target > 0:
            delta_12m = last_12m_avg - total_target
            delta_pct_12m = (delta_12m / total_target) * 100 if total_target > 0 else 0
            delta_text_12m = f"{delta_pct_12m:+.1f}%" if delta_12m != 0 else "On target"
            delta_color_12m = "inverse"
        else:
            delta_text_12m = None
            delta_color_12m = "normal"

        st.metric(
            label="📊 Last 12 Months Average",
            value=f"€{last_12m_avg:,.0f}",
            delta=delta_text_12m,
            delta_color=delta_color_12m,
            help="Average monthly expenses over the last 12 months",
        )

    with col2:
        # Available Budget - no delta
        st.metric(
            label="💰 Total Available Budget",
            value=f"€{total_available:,.0f}",
            help="Sum of all category available budgets",
        )

        # Suggested Budget - delta as percentage vs target
        if total_target > 0:
            delta_ratio = ((total_suggested - total_target) / total_target) * 100
            delta_text = f"{delta_ratio:+.1f}%"
            delta_color = "inverse"
        else:
            delta_text = None
            delta_color = "inverse"

        st.metric(
            label="💡 Total Suggested Budget",
            value=f"€{total_suggested:,.0f}",
            delta=delta_text,
            delta_color=delta_color,
            help="Sum of all category suggested budgets",
        )

    # Calculate global month range from filtered data
    global_month_range, earliest_date, latest_date = get_global_month_range(
        filtered_transactions_df, start_date, end_date
    )

    # Calculate global y-axis range for categories if enabled
    global_scale_enabled = st.session_state.get("global_scale_enabled", False)
    category_y_range = None
    if global_scale_enabled and filtered_category_names:
        category_y_range = calculate_global_y_range(
            filtered_transactions_df,
            budget_df,
            categories_data,
            "category",
            filtered_category_names,
        )

    # Category Analysis
    st.header("📊 Category Analysis")

    # Create a grid layout for the plots
    cols_per_row = 2
    for i in range(0, len(filtered_category_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(filtered_category_names):
                category_name = filtered_category_names[i + j]
                with col:
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
                        st.plotly_chart(category_fig, use_container_width=True)
                    else:
                        st.info(f"No data available for {category_name}")


if __name__ == "__main__":
    main()
