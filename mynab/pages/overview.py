import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from mynab.utils import (
    calculate_moving_average,
    calculate_forecast_trend,
    filter_data_by_date_range,
    get_global_month_range,
    get_excluded_groups,
    create_unified_plot,
    calculate_global_y_range,
    format_currency,
)

# Page configuration
st.set_page_config(
    page_title="Overview - YNAB Budget Dashboard", page_icon="📊", layout="wide"
)


def create_category_group_plot(
    group_name,
    transactions_df,
    budget_df,
    global_month_range,
    categories_data,
    y_range=None,
):
    """Create comprehensive plot for a single category group"""
    return create_unified_plot(
        group_name,
        transactions_df,
        budget_df,
        global_month_range,
        categories_data,
        "category_group",
        y_range,
    )


def create_comprehensive_plot(
    data_type,
    transactions_df,
    budget_df,
    global_month_range,
    y_range=None,
    target_goal=None,
):
    """Create comprehensive plot for total income, total expense, or total net income"""
    # Filter data based on type
    if data_type == "total_income":
        filtered_df = transactions_df[
            (transactions_df["category"] == "Inflow: Ready to Assign")
            & (transactions_df["payee_name"] != "Starting Balance")
        ].copy()
        color = "#2ca02c"  # Green for income
    elif data_type == "total_expense":
        # Include only transactions with a category group
        filtered_df = transactions_df[
            (transactions_df["category_group"].astype(str) != "nan")
            & (transactions_df["category_group"].astype(str) != "")
        ].copy()
        color = "#ff7f0e"  # Orange for expenses
    elif data_type == "total_net_income":
        # For net income, we need to calculate monthly income + monthly expenses
        # We'll process this differently in the aggregation step
        filtered_df = (
            transactions_df.copy()
        )  # Use all transactions for net income calculation
        color = "#1f77b4"  # Blue for net income

    if filtered_df.empty:
        return None

    # Aggregate transactions by month
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
    filtered_df["month"] = filtered_df["date"].dt.to_period("M")

    if data_type == "total_net_income":
        # For net income, calculate monthly income + monthly expenses
        # Filter for income transactions
        income_df = filtered_df[
            (filtered_df["category"] == "Inflow: Ready to Assign")
            & (filtered_df["payee_name"] != "Starting Balance")
        ]

        # Filter for expense transactions (with category group)
        expense_df = filtered_df[
            (filtered_df["category_group"].astype(str) != "nan")
            & (filtered_df["category_group"].astype(str) != "")
        ]

        # Group by month and sum
        monthly_income = income_df.groupby("month")["amount"].sum()
        monthly_expenses = expense_df.groupby("month")["amount"].sum()

        # Combine months and calculate net income
        all_months = pd.concat([monthly_income, monthly_expenses]).index.unique()
        monthly_net_income = pd.Series(index=all_months, dtype=float)

        for month in all_months:
            income_amount = monthly_income.get(month, 0)
            expense_amount = monthly_expenses.get(month, 0)
            # Since expenses are already negative in YNAB, we add them to income
            monthly_net_income[month] = income_amount + expense_amount

        # Convert to the expected format
        monthly_data = monthly_net_income.reset_index()
        monthly_data.columns = ["month", "amount"]
        monthly_data["month"] = monthly_data["month"].astype(str)
    else:
        # For income and expenses, use the original logic
        monthly_data = filtered_df.groupby("month")["amount"].sum().reset_index()
        monthly_data["month"] = monthly_data["month"].astype(str)

    # Use the global month range
    all_months_df = pd.DataFrame(
        {
            "month_date": global_month_range,
            "month": global_month_range.strftime("%Y-%m"),
        }
    )

    # Merge with actual data to include months with 0 values
    complete_monthly_data = all_months_df.merge(
        monthly_data[["month", "amount"]], on="month", how="left"
    ).fillna(0)

    # Sort by month date for proper ordering
    complete_monthly_data = complete_monthly_data.sort_values("month_date")

    # Create single comprehensive plot
    fig = go.Figure()

    # Bar chart for actual amounts - flip expenses to positive side
    y_values = complete_monthly_data["amount"]
    if data_type == "total_expense":
        y_values = abs(complete_monthly_data["amount"])

    # Pre-format hover text with currency formatting
    hover_text = [format_currency(val) for val in y_values]

    fig.add_trace(
        go.Bar(
            x=complete_monthly_data["month"],
            y=y_values,
            name=f"Actual {data_type.replace('_', ' ').title()}",
            marker_color=color,
            opacity=0.8,
            hovertemplate="<b>%{x}</b><br>"
            + f"{data_type.replace('_', ' ').title()}: %{{customdata}}<br>"
            + "<extra></extra>",
            customdata=hover_text,
        )
    )

    # Moving average line (if we have enough data) - flip expenses to positive side
    if len(complete_monthly_data) >= 3:
        # Use only the actual data (non-zero values) for calculations
        actual_data_mask = complete_monthly_data["amount"] != 0
        actual_data = complete_monthly_data[actual_data_mask].copy()

        if len(actual_data) >= 3:
            data_for_calculation = actual_data["amount"]
            if data_type == "total_expense":
                data_for_calculation = abs(actual_data["amount"])

            moving_avg = calculate_moving_average(data_for_calculation)

            # Pre-format hover text for moving average
            moving_avg_hover = [format_currency(val) for val in moving_avg]

            fig.add_trace(
                go.Scatter(
                    x=actual_data["month"],
                    y=moving_avg,
                    name="12-Month Moving Average",
                    line=dict(color="#1f77b4", width=2, dash="dash"),
                    mode="lines",
                    hovertemplate="<b>%{x}</b><br>"
                    + "12-Month Moving Average: %{customdata}<br>"
                    + "<extra></extra>",
                    customdata=moving_avg_hover,
                )
            )

            # Forecast trend line - flip expenses to positive side
            trend_line, forecast = calculate_forecast_trend(data_for_calculation)

            # Pre-format hover text for trend line
            trend_line_hover = [format_currency(val) for val in trend_line]

            fig.add_trace(
                go.Scatter(
                    x=actual_data["month"],
                    y=trend_line,
                    name="12-Month Forecast Trend",
                    line=dict(color="#d62728", width=2),
                    mode="lines",
                    hovertemplate="<b>%{x}</b><br>"
                    + "12-Month Forecast Trend: %{customdata}<br>"
                    + "<extra></extra>",
                    customdata=trend_line_hover,
                )
            )

            # Add forecast extension - flip expenses to positive side
            # Get the last month date from the actual data
            if len(actual_data) > 0:
                last_month_date = actual_data["month_date"].iloc[-1]
            else:
                last_month_date = complete_monthly_data["month_date"].iloc[-1]

            future_months = pd.date_range(
                start=last_month_date + pd.DateOffset(months=1),
                periods=3,
                freq="MS",
            )

            forecast_values = forecast
            if data_type == "total_expense":
                forecast_values = abs(forecast)

            # Pre-format hover text for forecast
            forecast_hover = [format_currency(val) for val in forecast_values]

            fig.add_trace(
                go.Scatter(
                    x=future_months.strftime("%Y-%m"),
                    y=forecast_values,
                    name="Forecast (Next 3 Months)",
                    line=dict(color="#d62728", width=2, dash="dot"),
                    mode="lines",
                    hovertemplate="<b>%{x}</b><br>"
                    + "Forecast: %{customdata}<br>"
                    + "<extra></extra>",
                    customdata=forecast_hover,
                )
            )

    # Add target goal line for total_expense if provided
    if data_type == "total_expense" and target_goal is not None and target_goal > 0:
        all_months = complete_monthly_data["month"].tolist()
        if "future_months" in locals():
            all_months += future_months.strftime("%Y-%m").tolist()
        target_hover = [format_currency(target_goal)] * len(all_months)
        fig.add_trace(
            go.Scatter(
                x=all_months,
                y=[target_goal] * len(all_months),
                name=f"Target Goal ({format_currency(target_goal)})",
                line=dict(color="#2ca02c", width=3, dash="solid"),
                mode="lines",
                hovertemplate="<b>%{x}</b><br>Target Goal: %{customdata}<br><extra></extra>",
                customdata=target_hover,
            )
        )

    # Update layout
    layout_update = {
        "height": 500,
        "showlegend": False,
        "hovermode": "x unified",
        "xaxis_title": "Month",
        "yaxis_title": "Amount (€)",
        "barmode": "overlay",
    }

    # Apply global y-axis range if provided
    if y_range is not None:
        layout_update["yaxis"] = {"range": y_range}

    fig.update_layout(**layout_update)

    return fig


# Main page content
st.markdown(
    '<h1 class="main-header">💰 YNAB Budget Dashboard</h1>', unsafe_allow_html=True
)

# Get data from session state
if not st.session_state.get("data_loaded", False):
    st.error("Data not loaded. Please go back to the main page.")
    st.stop()

# Get date range from session state
start_date = st.session_state.get("start_date")
end_date = st.session_state.get("end_date")

if start_date is None or end_date is None:
    st.error("Date range not set. Please configure the date range in the sidebar.")
    st.stop()

# Get data from session state
transactions_df = st.session_state.get("transactions_df")
budget_df = st.session_state.get("budget_df")
categories_data = st.session_state.get("categories_data")
selected_category_groups = st.session_state.get("selected_category_groups", [])

if transactions_df is None:
    st.error("No transaction data available.")
    st.stop()

# Filter out transactions with excluded category groups
excluded_groups = get_excluded_groups()
if transactions_df is not None:
    filtered_transactions_df = transactions_df[
        ~transactions_df["category_group"].isin(excluded_groups)
    ].copy()

    # Apply date filtering
    filtered_transactions_df = filter_data_by_date_range(
        filtered_transactions_df, start_date, end_date
    )
else:
    filtered_transactions_df = pd.DataFrame()

if not selected_category_groups:
    # Get all category group names (excluding the specified groups)
    category_groups = st.session_state.get("category_groups", {})
    category_group_names = sorted(
        [group for group in category_groups.keys() if group not in excluded_groups]
    )
    selected_category_groups = category_group_names

# Get selected categories from session state
selected_categories = st.session_state.get("selected_categories", [])

# Filter transactions to only include selected category groups and categories (but preserve income transactions)
if (
    isinstance(filtered_transactions_df, pd.DataFrame)
    and not filtered_transactions_df.empty
):
    # Keep income transactions (they don't have category groups)
    income_transactions = filtered_transactions_df[
        (filtered_transactions_df["category"] == "Inflow: Ready to Assign")
        & (filtered_transactions_df["payee_name"] != "Starting Balance")
    ].copy()

    # Filter expense transactions by selected category groups and categories
    expense_transactions = filtered_transactions_df[
        (filtered_transactions_df["category_group"].isin(selected_category_groups))
        & (filtered_transactions_df["category"].isin(selected_categories))
        & (filtered_transactions_df["category"] != "Inflow: Ready to Assign")
    ].copy()

    # Combine income and filtered expense transactions
    filtered_transactions_df = pd.concat(
        [income_transactions, expense_transactions], ignore_index=True
    )

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

# Calculate global month range from original data (before category filtering)
global_month_range, earliest_date, latest_date = get_global_month_range(
    transactions_df, start_date, end_date
)

# Calculate global y-axis ranges if enabled
global_scale_enabled = st.session_state.get("global_scale_enabled", False)
overview_y_range = None
category_group_y_range = None

if global_scale_enabled:
    # Calculate for overview plots
    overview_y_range = calculate_global_y_range(
        filtered_transactions_df,
        budget_df,
        categories_data,
        "overview",
    )

    # Calculate for category group plots
    if selected_category_groups:
        category_group_y_range = calculate_global_y_range(
            filtered_transactions_df,
            budget_df,
            categories_data,
            "category_group",
            selected_category_groups,
        )

# Display summary metrics - Row 1 (Totals)
col1, col2, col3 = st.columns(3)

with col1:
    total_income = filtered_transactions_df[
        (filtered_transactions_df["category"] == "Inflow: Ready to Assign")
        & (filtered_transactions_df["payee_name"] != "Starting Balance")
    ]["amount"].sum()
    st.metric("Total Income", format_currency(total_income))

with col2:
    # Include only transactions with a category group
    transactions_with_category = filtered_transactions_df[
        (filtered_transactions_df["category_group"].astype(str) != "nan")
        & (filtered_transactions_df["category_group"].astype(str) != "")
    ]
    total_expenses = transactions_with_category["amount"].sum()
    st.metric("Total Expenses", format_currency(total_expenses))

with col3:
    # Calculate total net income (income - expenses)
    # Since expenses are already negative in YNAB, we add them to income
    total_net_income = total_income + total_expenses
    st.metric("Total Net Income", format_currency(total_net_income))

# Display summary metrics - Row 2 (Averages)
col4, col5, col6 = st.columns(3)

with col4:
    # Calculate average monthly income
    if (
        isinstance(filtered_transactions_df, pd.DataFrame)
        and not filtered_transactions_df.empty
    ):
        # Filter for income transactions
        income_transactions = filtered_transactions_df[
            (filtered_transactions_df["category"] == "Inflow: Ready to Assign")
            & (filtered_transactions_df["payee_name"] != "Starting Balance")
        ].copy()

        if not income_transactions.empty:
            # Ensure date column is properly converted to datetime
            income_transactions = pd.DataFrame(income_transactions)
            income_transactions["date"] = pd.to_datetime(income_transactions["date"])
            # Group by month and calculate average
            monthly_income = income_transactions.groupby(
                income_transactions["date"].dt.to_period("M")
            )["amount"].sum()
            avg_monthly_income = monthly_income.mean()
            st.metric("Avg Monthly Income", format_currency(avg_monthly_income))
        else:
            st.metric("Avg Monthly Income", format_currency(0))
    else:
        st.metric("Avg Monthly Income", format_currency(0))

with col5:
    # Create a copy for calculations to avoid modifying the original
    if (
        isinstance(filtered_transactions_df, pd.DataFrame)
        and not filtered_transactions_df.empty
    ):
        calc_df = filtered_transactions_df.copy()
        calc_df["date"] = pd.to_datetime(calc_df["date"])
        # Include only transactions with a category group
        transactions_with_category = calc_df[
            (calc_df["category_group"].astype(str) != "nan")
            & (calc_df["category_group"].astype(str) != "")
        ]
        if not transactions_with_category.empty:
            # Ensure date column is properly converted to datetime
            transactions_with_category = pd.DataFrame(transactions_with_category)
            transactions_with_category["date"] = pd.to_datetime(
                transactions_with_category["date"]
            )
            # Group by month and calculate average
            monthly_expenses = transactions_with_category.groupby(
                transactions_with_category["date"].dt.to_period("M")
            )["amount"].sum()
            avg_monthly_expenses = monthly_expenses.mean()
            st.metric("Avg Monthly Expenses", format_currency(avg_monthly_expenses))
        else:
            st.metric("Avg Monthly Expenses", format_currency(0))
    else:
        st.metric("Avg Monthly Expenses", format_currency(0))

with col6:
    # Calculate average monthly net income
    if (
        isinstance(filtered_transactions_df, pd.DataFrame)
        and not filtered_transactions_df.empty
    ):
        # Get monthly income and expenses
        calc_df = filtered_transactions_df.copy()
        calc_df["date"] = pd.to_datetime(calc_df["date"])

        # Monthly income
        income_transactions = calc_df[
            (calc_df["category"] == "Inflow: Ready to Assign")
            & (calc_df["payee_name"] != "Starting Balance")
        ]

        # Monthly expenses
        expense_transactions = calc_df[
            (calc_df["category_group"].astype(str) != "nan")
            & (calc_df["category_group"].astype(str) != "")
        ]

        if not income_transactions.empty or not expense_transactions.empty:
            # Group by month and calculate net income
            income_transactions = pd.DataFrame(income_transactions)
            expense_transactions = pd.DataFrame(expense_transactions)
            monthly_income = income_transactions.groupby(
                income_transactions["date"].dt.to_period("M")
            )["amount"].sum()
            monthly_expenses = expense_transactions.groupby(
                expense_transactions["date"].dt.to_period("M")
            )["amount"].sum()

            # Combine months and calculate net income
            all_months = monthly_income.index.union(monthly_expenses.index)
            monthly_net_income = pd.Series(index=all_months, dtype=float)

            for month in all_months:
                income_amount = monthly_income.get(month, 0) or 0
                expense_amount = monthly_expenses.get(month, 0) or 0
                # Since expenses are already negative in YNAB, we add them to income
                net_income = income_amount + expense_amount
                monthly_net_income[month] = net_income

            avg_monthly_net_income = monthly_net_income.mean()
            st.metric("Avg Monthly Net Income", format_currency(avg_monthly_net_income))
        else:
            st.metric("Avg Monthly Net Income", format_currency(0))
    else:
        st.metric("Avg Monthly Net Income", format_currency(0))

st.markdown("---")

# Overview Analysis
st.header("📊 Overview Analysis")

# Create three plots in a row for overview
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
        st.plotly_chart(income_fig, use_container_width=True)
    else:
        st.info("No income data available")

with col2:
    st.subheader("💸 Total Expenses")
    # Calculate total target goal for selected category groups
    # (before rendering the Total Expenses plot)
    total_target_goal = sum(
        cat.get("target_amount", 0) or 0
        for cat in categories_data
        if cat["group"] in selected_category_groups and cat.get("target_amount")
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
        st.plotly_chart(expense_fig, use_container_width=True)
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
        st.plotly_chart(net_income_fig, use_container_width=True)
    else:
        st.info("No net income data available")

st.markdown("---")

# Category Group Analysis
st.header("📊 Category Group Analysis")

# Create a grid layout for the plots
cols_per_row = 2
for i in range(0, len(selected_category_groups), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        if i + j < len(selected_category_groups):
            group_name = selected_category_groups[i + j]
            with col:
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
                    st.plotly_chart(group_fig, use_container_width=True)
                else:
                    st.info(f"No data available for {group_name}")
