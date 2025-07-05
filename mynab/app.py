import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
import os
from mynab.utils import (
    get_ynab_data, process_categories_data, process_transactions_data, process_months_data,
    calculate_moving_average, calculate_forecast_trend, calculate_category_group_averages,
    calculate_category_group_available_budget, filter_data_by_date_range, safe_strftime,
    get_default_date_range, get_global_month_range, get_excluded_groups, get_default_category_groups
)

# Page configuration
st.set_page_config(
    page_title="YNAB Budget Dashboard",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
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
""", unsafe_allow_html=True)



def create_category_group_plot(group_name, transactions_df, budget_df, global_month_range, categories_data):
    """Create comprehensive plot for a single category group"""
    # Filter data for this category group
    group_transactions = transactions_df[transactions_df['category_group'] == group_name].copy()
    
    if group_transactions.empty:
        return None
    
    # Calculate total target goal for this category group
    group_target_amount = 0
    categories_with_targets = []
    
    for cat in categories_data:
        if cat['group'] == group_name and cat.get('target_amount') is not None:
            group_target_amount += cat['target_amount']
            categories_with_targets.append(cat)
    
    # Calculate average metrics
    avg_12_months = calculate_category_group_averages(group_name, transactions_df, 12)
    
    # Calculate available budget for this category group
    available_budget = calculate_category_group_available_budget(group_name, budget_df)
    
    # Create metrics row - First row
    col1, col2 = st.columns(2)
    
    with col1:
        if group_target_amount > 0:
            # Calculate delta for 12-month average
            delta_12m = avg_12_months - group_target_amount
            delta_color_12m = "normal"
            delta_text_12m = f"€{delta_12m:,.0f} vs target" if delta_12m != 0 else "On target"
            
            st.metric(
                label="🎯 Target Budget",
                value=f"€{group_target_amount:,.0f}",
                # delta=delta_text_12m,
                # delta_color=delta_color_12m
            )
        else:
            st.metric(
                label="🎯 Target Budget",
                value="No target set"
            )
    
    with col2:        
        st.metric(
            label="💰 Available Budget",
            value=f"€{available_budget:,.0f}",
        )
    
    # Create metrics row - Second row
    col3, col4 = st.columns(2)
    
    with col3:
        if group_target_amount > 0:
            # Calculate delta for 12-month average as percentage
            delta_12m = avg_12_months - group_target_amount
            delta_pct_12m = (delta_12m / group_target_amount) * 100 if group_target_amount > 0 else 0
            delta_color_12m = "inverse"
            delta_text_12m = f"{delta_pct_12m:+.1f}%" if delta_12m != 0 else "On target"
        else:
            delta_text_12m = None
            delta_color_12m = "normal"
        
        st.metric(
            label="📊 Last 12 Months Avg",
            value=f"€{avg_12_months:,.0f}",
            delta=delta_text_12m,
            delta_color=delta_color_12m
        )
    
    with col4:
        # Calculate suggested budget: -(available_budget - (avg_12_months*12))/12
        suggested_budget = -(available_budget - (avg_12_months * 12)) / 12
        
        # Calculate delta as difference ratio to target budget
        if group_target_amount > 0:
            delta_ratio = ((suggested_budget - group_target_amount) / group_target_amount) * 100
            delta_text = f"{delta_ratio:+.1f}%"
            delta_color = "inverse"
        else:
            delta_text = None
            delta_color = "inverse"
        
        st.metric(
            label="💡 Suggested Budget",
            value=f"€{suggested_budget:,.0f}",
            delta=delta_text,
            delta_color=delta_color
        )
    
    # Aggregate transactions by month
    group_transactions['date'] = pd.to_datetime(group_transactions['date'])
    group_transactions['month'] = group_transactions['date'].dt.to_period('M')
    monthly_expenses = group_transactions.groupby('month')['amount'].sum().reset_index()
    monthly_expenses['month'] = monthly_expenses['month'].astype(str)
    
    # Create single comprehensive plot
    fig = go.Figure()
    
    if not monthly_expenses.empty:
        # Sort by month for proper ordering
        monthly_expenses = monthly_expenses.sort_values('month')
        monthly_expenses['month_date'] = pd.to_datetime(monthly_expenses['month'])
        monthly_expenses = monthly_expenses.sort_values('month_date')
        
        # Use the global month range instead of group-specific range
        all_months_df = pd.DataFrame({
            'month_date': global_month_range,
            'month': global_month_range.strftime('%Y-%m')
        })
        
        # Merge with actual data to include months with 0 expenses
        complete_monthly_data = all_months_df.merge(
            monthly_expenses[['month', 'amount']], 
            on='month', 
            how='left'
        ).fillna(0)
        
        # Sort by month date for proper ordering
        complete_monthly_data = complete_monthly_data.sort_values('month_date')
        
        # Bar chart for actual expenses (including 0 values) - flipped to positive side
        fig.add_trace(
            go.Bar(
                x=complete_monthly_data['month'],
                y=abs(complete_monthly_data['amount']),
                name='Actual Expenses',
                marker_color='#ff7f0e',
                opacity=0.8
            )
        )
        
        # Moving average line (if we have enough data) - flipped to positive side
        if len(complete_monthly_data) >= 3:
            moving_avg = calculate_moving_average(abs(complete_monthly_data['amount']))
            
            fig.add_trace(
                go.Scatter(
                    x=complete_monthly_data['month'],
                    y=moving_avg,
                    name='12-Month Moving Average',
                    line=dict(color='#1f77b4', width=2, dash='dash'),
                    mode='lines'
                )
            )
            
            # Forecast trend line - flipped to positive side
            trend_line, forecast = calculate_forecast_trend(abs(complete_monthly_data['amount']))
            
            fig.add_trace(
                go.Scatter(
                    x=complete_monthly_data['month'],
                    y=trend_line,
                    name='12-Month Forecast Trend',
                    line=dict(color='#d62728', width=2),
                    mode='lines'
                )
            )
            
            # Add forecast extension - flipped to positive side
            future_months = pd.date_range(
                start=complete_monthly_data['month_date'].iloc[-1] + pd.DateOffset(months=1),
                periods=3,
                freq='MS'
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_months.strftime('%Y-%m'),
                    y=abs(forecast),
                    name='Forecast (Next 3 Months)',
                    line=dict(color='#d62728', width=2, dash='dot'),
                    mode='lines'
                )
            )
        
        # Add target goal line if available
        if group_target_amount > 0:
            # Create a horizontal line across all months
            all_months = complete_monthly_data['month'].tolist()
            if len(complete_monthly_data) >= 3 and 'future_months' in locals():
                # Include future months in the target line
                future_month_strs = future_months.strftime('%Y-%m').tolist()
                all_months.extend(future_month_strs)
            
            fig.add_trace(
                go.Scatter(
                    x=all_months,
                    y=[group_target_amount] * len(all_months),
                    name=f'Target Goal (€{group_target_amount:,.0f})',
                    line=dict(color='#2ca02c', width=3, dash='solid'),
                    mode='lines'
                )
            )
    
    # Update layout
    fig.update_layout(
        # title=f'{group_name} - Comprehensive Analysis{target_info}',
        height=400,
        showlegend=False,
        hovermode='x unified',
        xaxis_title='Month',
        yaxis_title='Amount (€)',
        barmode='overlay'
    )
    
    return fig

def create_comprehensive_plot(data_type, transactions_df, budget_df, global_month_range):
    """Create comprehensive plot for total income, total expense, or total net income"""
    # Filter data based on type
    if data_type == 'total_income':
        filtered_df = transactions_df[
            (transactions_df['category'] == 'Inflow: Ready to Assign') & 
            (transactions_df['payee_name'] != 'Starting Balance')
        ].copy()
        color = '#2ca02c'  # Green for income
    elif data_type == 'total_expense':
        # Include only transactions with a category group
        filtered_df = transactions_df[
            (transactions_df['category_group'].astype(str) != 'nan') & 
            (transactions_df['category_group'].astype(str) != '')
        ].copy()
        color = '#ff7f0e'  # Orange for expenses
    elif data_type == 'total_net_income':
        # For net income, we need to calculate monthly income + monthly expenses
        # We'll process this differently in the aggregation step
        filtered_df = transactions_df.copy()  # Use all transactions for net income calculation
        color = '#1f77b4'  # Blue for net income
    
    if filtered_df.empty:
        return None
    
    # Aggregate transactions by month
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    filtered_df['month'] = filtered_df['date'].dt.to_period('M')
    
    if data_type == 'total_net_income':
        # For net income, calculate monthly income + monthly expenses
        # Filter for income transactions
        income_df = filtered_df[
            (filtered_df['category'] == 'Inflow: Ready to Assign') & 
            (filtered_df['payee_name'] != 'Starting Balance')
        ]
        
        # Filter for expense transactions (with category group)
        expense_df = filtered_df[
            (filtered_df['category_group'].astype(str) != 'nan') & 
            (filtered_df['category_group'].astype(str) != '')
        ]
        
        # Group by month and sum
        monthly_income = income_df.groupby('month')['amount'].sum()
        monthly_expenses = expense_df.groupby('month')['amount'].sum()
        
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
        monthly_data.columns = ['month', 'amount']
        monthly_data['month'] = monthly_data['month'].astype(str)
    else:
        # For income and expenses, use the original logic
        monthly_data = filtered_df.groupby('month')['amount'].sum().reset_index()
        monthly_data['month'] = monthly_data['month'].astype(str)
    
    # Use the global month range
    all_months_df = pd.DataFrame({
        'month_date': global_month_range,
        'month': global_month_range.strftime('%Y-%m')
    })
    
    # Merge with actual data to include months with 0 values
    complete_monthly_data = all_months_df.merge(
        monthly_data[['month', 'amount']], 
        on='month', 
        how='left'
    ).fillna(0)
    
    # Sort by month date for proper ordering
    complete_monthly_data = complete_monthly_data.sort_values('month_date')
    
    # Create single comprehensive plot
    fig = go.Figure()
    
    # Bar chart for actual amounts - flip expenses to positive side
    y_values = complete_monthly_data['amount']
    if data_type == 'total_expense':
        y_values = abs(complete_monthly_data['amount'])
    
    fig.add_trace(
        go.Bar(
            x=complete_monthly_data['month'],
            y=y_values,
            name=f'Actual {data_type.replace("_", " ").title()}',
            marker_color=color,
            opacity=0.8
        )
    )
    
    # Moving average line (if we have enough data) - flip expenses to positive side
    if len(complete_monthly_data) >= 3:
        data_for_calculation = complete_monthly_data['amount']
        if data_type == 'total_expense':
            data_for_calculation = abs(complete_monthly_data['amount'])
        
        moving_avg = calculate_moving_average(data_for_calculation)
        
        fig.add_trace(
            go.Scatter(
                x=complete_monthly_data['month'],
                y=moving_avg,
                name='12-Month Moving Average',
                line=dict(color='#1f77b4', width=2, dash='dash'),
                mode='lines'
            )
        )
        
        # Forecast trend line - flip expenses to positive side
        trend_line, forecast = calculate_forecast_trend(data_for_calculation)
        
        fig.add_trace(
            go.Scatter(
                x=complete_monthly_data['month'],
                y=trend_line,
                name='12-Month Forecast Trend',
                line=dict(color='#d62728', width=2),
                mode='lines'
            )
        )
        
        # Add forecast extension - flip expenses to positive side
        future_months = pd.date_range(
            start=complete_monthly_data['month_date'].iloc[-1] + pd.DateOffset(months=1),
            periods=3,
            freq='MS'
        )
        
        forecast_values = forecast
        if data_type == 'total_expense':
            forecast_values = abs(forecast)
        
        fig.add_trace(
            go.Scatter(
                x=future_months.strftime('%Y-%m'),
                y=forecast_values,
                name='Forecast (Next 3 Months)',
                line=dict(color='#d62728', width=2, dash='dot'),
                mode='lines'
            )
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        hovermode='x unified',
        xaxis_title='Month',
        yaxis_title='Amount (€)',
        barmode='overlay'
    )
    
    return fig



def main():
    st.markdown('<h1 class="main-header">💰 YNAB Budget Dashboard</h1>', unsafe_allow_html=True)
    
    # Get API token from environment
    api_token = os.getenv('YNAB_API_KEY')
    
    if not api_token:
        st.error("YNAB API key not found in environment variables.")
        st.info("""
        Please set your YNAB API key in the .env file:
        1. Edit the .env file in your project directory
        2. Replace 'your_ynab_api_token_here' with your actual API token
        3. Get your API token from YNAB Account Settings > Developer Settings
        """)
        return
    
    # Fetch data
    with st.spinner("Fetching data from YNAB..."):
        budget_id, budget_name, categories_response, transactions_response, months_response = get_ynab_data(api_token)
    
    if not budget_id:
        return
    
    st.success(f"Connected to budget: **{budget_name}**")
    
    # Process data
    categories_data, category_groups = process_categories_data(categories_response)
    transactions_df = process_transactions_data(transactions_response, categories_data)
    budget_df = process_months_data(months_response, categories_data)
    
    # Filter out transactions with excluded category groups
    excluded_groups = get_excluded_groups()
    filtered_transactions_df = transactions_df[~transactions_df['category_group'].isin(excluded_groups)].copy()
    
    # Sidebar date picker
    st.sidebar.header("📅 Date Range Filter")
    
    # Get default date range
    default_start_date, default_end_date = get_default_date_range()
    
    # Get the actual date range from the data
    if isinstance(filtered_transactions_df, pd.DataFrame) and not filtered_transactions_df.empty:
        filtered_transactions_df['date'] = pd.to_datetime(filtered_transactions_df['date'])
        data_start_date = filtered_transactions_df['date'].min().date()
        data_end_date = filtered_transactions_df['date'].max().date()
        
        # Use data range as defaults if available, but cap end date to last day of prior month
        default_start_date = data_start_date
        default_end_date = min(data_end_date, default_end_date)
    
    # Date picker in sidebar
    today = date.today()
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        min_value=date(2010, 1, 1),
        max_value=today,
        help="Select the start date for filtering data"
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end_date,
        min_value=date(2010, 1, 1),
        max_value=today,
        help="Select the end date for filtering data"
    )
    
    # Validate date range
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date!")
        return
    
    # Apply date filtering
    filtered_transactions_df = filter_data_by_date_range(filtered_transactions_df, start_date, end_date)
    
    # Category Group Filter in sidebar
    st.sidebar.header("📊 Category Group Filter")
    
    # Get all category group names (excluding the specified groups)
    excluded_groups = get_excluded_groups()
    category_group_names = sorted([group for group in category_groups.keys() if group not in excluded_groups])
    
    # Set default values for multiselect
    default_groups = get_default_category_groups()
    # Filter default groups to only include those that exist in the data
    available_defaults = [group for group in default_groups if group in category_group_names]
    
    selected_category_groups = st.sidebar.multiselect(
        "Select category groups to include:",
        options=category_group_names,
        default=available_defaults,
        help="Choose which category groups to include in the analysis. Leave empty to show all groups."
    )
    
    # If no groups are selected, show all groups
    if not selected_category_groups:
        selected_category_groups = category_group_names
    
    # Filter transactions to only include selected category groups (but preserve income transactions)
    if isinstance(filtered_transactions_df, pd.DataFrame) and not filtered_transactions_df.empty:
        # Keep income transactions (they don't have category groups)
        income_transactions = filtered_transactions_df[
            (filtered_transactions_df['category'] == 'Inflow: Ready to Assign') & 
            (filtered_transactions_df['payee_name'] != 'Starting Balance')
        ].copy()
        
        # Filter expense transactions by selected category groups
        expense_transactions = filtered_transactions_df[
            (filtered_transactions_df['category_group'].isin(selected_category_groups)) &
            (filtered_transactions_df['category'] != 'Inflow: Ready to Assign')
        ].copy()
        
        # Combine income and filtered expense transactions
        filtered_transactions_df = pd.concat([income_transactions, expense_transactions], ignore_index=True)
    
    # Filter budget data to only include selected category groups
    if isinstance(budget_df, pd.DataFrame) and not budget_df.empty:
        budget_df = budget_df[budget_df['category_group'].isin(selected_category_groups)].copy()
    
    # Show date range info in sidebar
    start_str = safe_strftime(start_date)
    end_str = safe_strftime(end_date)
    st.sidebar.info(f"Date range: {start_str} to {end_str}")
    st.sidebar.info(f"Selected groups: {len(selected_category_groups)} of {len(category_group_names)}")
    # Count transactions for sidebar info
    filtered_df = pd.DataFrame(filtered_transactions_df)
    income_count = len(filtered_df[
        (filtered_df['category'] == 'Inflow: Ready to Assign') & 
        (filtered_df['payee_name'] != 'Starting Balance')
    ])
    expense_count = len(filtered_df[
        (filtered_df['category_group'].isin(selected_category_groups)) &
        (filtered_df['category'] != 'Inflow: Ready to Assign')
    ])
    st.sidebar.info(f"📈 {income_count} income + {expense_count} expense transactions")
    
    # Calculate global month range from original data (before category filtering)
    global_month_range, earliest_date, latest_date = get_global_month_range(transactions_df, start_date, end_date)
    
    # Display summary metrics - Row 1 (Totals)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_income = filtered_transactions_df[
            (filtered_transactions_df['category'] == 'Inflow: Ready to Assign') & 
            (filtered_transactions_df['payee_name'] != 'Starting Balance')
        ]['amount'].sum()
        st.metric("Total Income", f"€{total_income:,.2f}")
    
    with col2:
        # Include only transactions with a category group
        transactions_with_category = filtered_transactions_df[
            (filtered_transactions_df['category_group'].astype(str) != 'nan') & 
            (filtered_transactions_df['category_group'].astype(str) != '')
        ]
        total_expenses = transactions_with_category['amount'].sum()
        st.metric("Total Expenses", f"€{total_expenses:,.2f}")
    
    with col3:
        # Calculate total net income (income - expenses)
        # Since expenses are already negative in YNAB, we add them to income
        total_net_income = total_income + total_expenses
        st.metric("Total Net Income", f"€{total_net_income:,.2f}")
    
    # Display summary metrics - Row 2 (Averages)
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Calculate average monthly income
        if isinstance(filtered_transactions_df, pd.DataFrame) and not filtered_transactions_df.empty:
            # Filter for income transactions
            income_transactions = filtered_transactions_df[
                (filtered_transactions_df['category'] == 'Inflow: Ready to Assign') & 
                (filtered_transactions_df['payee_name'] != 'Starting Balance')
            ].copy()
            
            if not income_transactions.empty:
                # Ensure date column is properly converted to datetime
                income_transactions = pd.DataFrame(income_transactions)
                income_transactions['date'] = pd.to_datetime(income_transactions['date'])
                # Group by month and calculate average
                monthly_income = income_transactions.groupby(income_transactions['date'].dt.to_period('M'))['amount'].sum()
                avg_monthly_income = monthly_income.mean()
                st.metric("Avg Monthly Income", f"€{avg_monthly_income:,.2f}")
            else:
                st.metric("Avg Monthly Income", "€0.00")
        else:
            st.metric("Avg Monthly Income", "€0.00")
    
    with col5:
        # Create a copy for calculations to avoid modifying the original
        if isinstance(filtered_transactions_df, pd.DataFrame) and not filtered_transactions_df.empty:
            calc_df = filtered_transactions_df.copy()
            calc_df['date'] = pd.to_datetime(calc_df['date'])
            # Include only transactions with a category group
            transactions_with_category = calc_df[
                (calc_df['category_group'].astype(str) != 'nan') & 
                (calc_df['category_group'].astype(str) != '')
            ]
            if not transactions_with_category.empty:
                # Ensure date column is properly converted to datetime
                transactions_with_category = pd.DataFrame(transactions_with_category)
                transactions_with_category['date'] = pd.to_datetime(transactions_with_category['date'])
                # Group by month and calculate average
                monthly_expenses = transactions_with_category.groupby(transactions_with_category['date'].dt.to_period('M'))['amount'].sum()
                avg_monthly_expenses = monthly_expenses.mean()
                st.metric("Avg Monthly Expenses", f"€{avg_monthly_expenses:,.2f}")
            else:
                st.metric("Avg Monthly Expenses", "€0.00")
        else:
            st.metric("Avg Monthly Expenses", "€0.00")
    
    with col6:
        # Calculate average monthly net income
        if isinstance(filtered_transactions_df, pd.DataFrame) and not filtered_transactions_df.empty:
            # Get monthly income and expenses
            calc_df = filtered_transactions_df.copy()
            calc_df['date'] = pd.to_datetime(calc_df['date'])
            
            # Monthly income
            income_transactions = calc_df[
                (calc_df['category'] == 'Inflow: Ready to Assign') & 
                (calc_df['payee_name'] != 'Starting Balance')
            ]
            
            # Monthly expenses
            expense_transactions = calc_df[
                (calc_df['category_group'].astype(str) != 'nan') & 
                (calc_df['category_group'].astype(str) != '')
            ]
            
            if not income_transactions.empty or not expense_transactions.empty:
                # Group by month and calculate net income
                income_transactions = pd.DataFrame(income_transactions)
                expense_transactions = pd.DataFrame(expense_transactions)
                monthly_income = income_transactions.groupby(income_transactions['date'].dt.to_period('M'))['amount'].sum()
                monthly_expenses = expense_transactions.groupby(expense_transactions['date'].dt.to_period('M'))['amount'].sum()
                
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
                st.metric("Avg Monthly Net Income", f"€{avg_monthly_net_income:,.2f}")
            else:
                st.metric("Avg Monthly Net Income", "€0.00")
        else:
            st.metric("Avg Monthly Net Income", "€0.00")
    
    st.markdown("---")
    
    # Overview Analysis
    st.header("📊 Overview Analysis")
    
    # Create three plots in a row for overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 Total Income")
        income_fig = create_comprehensive_plot('total_income', filtered_transactions_df, budget_df, global_month_range)
        if income_fig:
            st.plotly_chart(income_fig, use_container_width=True)
        else:
            st.info("No income data available")
    
    with col2:
        st.subheader("💸 Total Expenses")
        expense_fig = create_comprehensive_plot('total_expense', filtered_transactions_df, budget_df, global_month_range)
        if expense_fig:
            st.plotly_chart(expense_fig, use_container_width=True)
        else:
            st.info("No expense data available")
    
    with col3:
        st.subheader("📈 Monthly Net Income")
        net_income_fig = create_comprehensive_plot('total_net_income', filtered_transactions_df, budget_df, global_month_range)
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
                    group_fig = create_category_group_plot(group_name, filtered_transactions_df, budget_df, global_month_range, categories_data)
                    if group_fig:
                        st.plotly_chart(group_fig, use_container_width=True)
                    else:
                        st.info(f"No data available for {group_name}")
    
    st.markdown("---")
    
    # Data tables
    st.header("📄 Raw Data")
    
    tab1, tab2, tab3 = st.tabs(["Transactions", "Budget Data", "Categories"])
    
    with tab1:
        # Add category filter
        if isinstance(filtered_transactions_df, pd.DataFrame) and not filtered_transactions_df.empty and 'category' in filtered_transactions_df.columns:
            # Get unique categories for the selectbox (from filtered data)
            unique_categories = ['All Categories'] + sorted(filtered_transactions_df['category'].unique().tolist())
            selected_category = st.selectbox(
                "Filter by Category:",
                options=unique_categories,
                index=0
            )
            
            # Filter dataframe based on selection
            if selected_category == 'All Categories':
                display_df = filtered_transactions_df
            else:
                display_df = filtered_transactions_df[filtered_transactions_df['category'] == selected_category]
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.dataframe(filtered_transactions_df, use_container_width=True)
    
    with tab2:
        if not budget_df.empty:
            st.dataframe(budget_df, use_container_width=True)
        else:
            st.info("No budget data available - using transaction data for analysis")
    
    with tab3:
        categories_df = pd.DataFrame(categories_data)
        
        # Add a summary of categories with targets
        if not categories_df.empty and 'target_amount' in categories_df.columns:
            categories_with_targets = categories_df[categories_df['target_amount'].notna()]
            if not categories_with_targets.empty:
                st.subheader("🎯 Categories with Targets")
                st.info(f"Found {len(categories_with_targets)} categories with target values")
                
                # Display target summary
                target_summary = pd.DataFrame(categories_with_targets[['name', 'group', 'target_amount', 'target_type', 'target_date']].copy())
                target_summary = target_summary.rename(columns={
                    'name': 'Category',
                    'group': 'Category Group',
                    'target_amount': 'Target Amount (€)',
                    'target_type': 'Target Type',
                    'target_date': 'Target Date'
                })
                st.dataframe(target_summary, use_container_width=True)
                
                st.markdown("---")
        
        st.subheader("📋 All Categories")
        st.dataframe(categories_df, use_container_width=True)

if __name__ == "__main__":
    main() 