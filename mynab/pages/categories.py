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
    calculate_moving_average, calculate_forecast_trend, calculate_category_averages,
    calculate_category_available_budget, filter_data_by_date_range, safe_strftime,
    get_default_date_range, get_global_month_range, get_excluded_groups, get_default_categories,
    get_default_category_groups
)

# Page configuration
st.set_page_config(
    page_title="Individual Category Analysis - YNAB Dashboard",
    page_icon="📋",
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



def create_category_plot(category_name, transactions_df, budget_df, global_month_range, categories_data):
    """Create comprehensive plot for a single category with enhanced metrics"""
    # Filter data for this category
    cat_transactions = transactions_df[transactions_df['category'] == category_name].copy()
    
    # Get budget data for this category
    cat_budget = pd.DataFrame()
    if not budget_df.empty and 'category' in budget_df.columns:
        cat_budget = budget_df[budget_df['category'] == category_name].copy()
    
    if cat_transactions.empty and cat_budget.empty:
        return None
    
    # Find target goal for this category
    target_amount = None
    for cat in categories_data:
        if cat['name'] == category_name and cat.get('target_amount') is not None:
            target_amount = cat['target_amount']
            break
    
    # Calculate average metrics
    avg_12_months = calculate_category_averages(category_name, transactions_df, 12)
    
    # Calculate available budget for this category
    available_budget = calculate_category_available_budget(category_name, budget_df)
    
    # Create metrics row - First row
    col1, col2 = st.columns(2)
    
    with col1:
        if target_amount is not None and target_amount > 0:
            # Calculate delta for 12-month average
            delta_12m = avg_12_months - target_amount
            delta_color_12m = "normal"
            delta_text_12m = f"€{delta_12m:,.0f} vs target" if delta_12m != 0 else "On target"
            
            st.metric(
                label="🎯 Target Budget",
                value=f"€{target_amount:,.0f}",
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
        if target_amount is not None and target_amount > 0:
            # Calculate delta for 12-month average as percentage
            delta_12m = avg_12_months - target_amount
            delta_pct_12m = (delta_12m / target_amount) * 100 if target_amount > 0 else 0
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
        if target_amount is not None and target_amount > 0:
            delta_ratio = ((suggested_budget - target_amount) / target_amount) * 100
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
    cat_transactions['date'] = pd.to_datetime(cat_transactions['date'])
    cat_transactions['month'] = cat_transactions['date'].dt.to_period('M')
    monthly_expenses = cat_transactions.groupby('month')['amount'].sum().reset_index()
    monthly_expenses['month'] = monthly_expenses['month'].astype(str)
    
    # Create single comprehensive plot
    fig = go.Figure()
    
    if not monthly_expenses.empty:
        # Sort by month for proper ordering
        monthly_expenses = monthly_expenses.sort_values('month')
        monthly_expenses['month_date'] = pd.to_datetime(monthly_expenses['month'])
        monthly_expenses = monthly_expenses.sort_values('month_date')
        
        # Use the global month range instead of category-specific range
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
        if target_amount is not None and target_amount > 0:
            # Create a horizontal line across all months
            all_months = complete_monthly_data['month'].tolist()
            if len(complete_monthly_data) >= 3 and 'future_months' in locals():
                # Include future months in the target line
                future_month_strs = future_months.strftime('%Y-%m').tolist()
                all_months.extend(future_month_strs)
            
            fig.add_trace(
                go.Scatter(
                    x=all_months,
                    y=[target_amount] * len(all_months),
                    name=f'Target Goal (€{target_amount:,.0f})',
                    line=dict(color='#2ca02c', width=3, dash='solid'),
                    mode='lines'
                )
            )
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=False,
        hovermode='x unified',
        xaxis_title='Month',
        yaxis_title='Amount (€)',
        barmode='overlay'
    )
    
    return fig



def main():
    st.markdown('<h1 class="main-header">📋 Individual Category Analysis</h1>', unsafe_allow_html=True)
    
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
    
    # Sidebar filters
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
    
    # Filter categories based on selected groups
    filtered_categories_data = [cat for cat in categories_data if cat['group'] in selected_category_groups]
    filtered_category_names = [cat['name'] for cat in filtered_categories_data]
    
    # Filter transactions to only include selected category groups
    if isinstance(filtered_transactions_df, pd.DataFrame) and not filtered_transactions_df.empty:
        filtered_transactions_df = filtered_transactions_df[filtered_transactions_df['category_group'].isin(selected_category_groups)].copy()
    
    # Filter budget data to only include selected category groups
    if isinstance(budget_df, pd.DataFrame) and not budget_df.empty:
        budget_df = budget_df[budget_df['category_group'].isin(selected_category_groups)].copy()
    
    # Show date range info in sidebar
    start_str = safe_strftime(start_date)
    end_str = safe_strftime(end_date)
    st.sidebar.info(f"Date range: {start_str} to {end_str}")
    st.sidebar.info(f"Selected groups: {len(selected_category_groups)} of {len(category_group_names)}")
    
    # Calculate summary metrics
    def calculate_summary_metrics(categories_data, budget_df, filtered_transactions_df):
        """Calculate summary metrics for all categories"""
        # Total target budget
        total_target_budget = sum(
            cat.get('target_amount', 0) or 0 
            for cat in categories_data 
            if cat.get('target_amount') is not None
        )
        
        # Total available budget
        total_available_budget = budget_df['available'].sum() if not budget_df.empty else 0
        
        # Last 12 months average for all transactions
        if not filtered_transactions_df.empty:
            # Use all transactions (both income and expenses)
            all_transactions_df = filtered_transactions_df.copy()
            all_transactions_df['date'] = pd.to_datetime(all_transactions_df['date'])
            all_transactions_df['month'] = all_transactions_df['date'].dt.to_period('M')
            monthly_totals = all_transactions_df.groupby('month')['amount'].sum()
            if len(monthly_totals) >= 12:
                last_12_months_avg = abs(monthly_totals.tail(12).mean())
            else:
                last_12_months_avg = abs(monthly_totals.mean()) if not monthly_totals.empty else 0
        else:
            last_12_months_avg = 0
        
        # Sum of suggested budgets
        total_suggested_budget = 0
        for cat in categories_data:
            category_name = cat['name']
            avg_12_months = calculate_category_averages(category_name, filtered_transactions_df, 12)
            available_budget = calculate_category_available_budget(category_name, budget_df)
            suggested_budget = -(available_budget - (avg_12_months * 12)) / 12
            total_suggested_budget += suggested_budget
        
        return total_target_budget, total_available_budget, last_12_months_avg, total_suggested_budget
    
    # Calculate summary metrics
    total_target, total_available, last_12m_avg, total_suggested = calculate_summary_metrics(
        filtered_categories_data, budget_df, filtered_transactions_df
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
            help="Sum of all category target budgets"
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
            help="Average monthly expenses over the last 12 months"
        )
    
    with col2:
        # Available Budget - no delta
        st.metric(
            label="💰 Total Available Budget",
            value=f"€{total_available:,.0f}",
            help="Sum of all category available budgets"
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
            help="Sum of all category suggested budgets"
        )
    
    # Calculate global month range from filtered data
    global_month_range, earliest_date, latest_date = get_global_month_range(filtered_transactions_df, start_date, end_date)
    
    # Display date range info
    start_str = safe_strftime(earliest_date)
    end_str = safe_strftime(latest_date)
    st.info(f"Date range: {start_str} to {end_str}")
    
    # Category selection
    st.header("📊 Category Analysis")
    
    # Set default values for multiselect
    default_categories = get_default_categories()
    # Filter default categories to only include those that exist in the filtered data
    available_defaults = [cat for cat in default_categories if cat in filtered_category_names]
    
    selected_categories = st.multiselect(
        "Select categories to display:",
        options=filtered_category_names,
        default=available_defaults,
        help="Choose which categories to include in the analysis. Leave empty to show all categories from selected groups."
    )
    
    # If no categories are selected, show all categories from selected groups
    if not selected_categories:
        selected_categories = filtered_category_names
    
    # Create a grid layout for the plots
    cols_per_row = 2
    for i in range(0, len(selected_categories), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(selected_categories):
                category_name = selected_categories[i + j]
                with col:
                    st.subheader(category_name)
                    category_fig = create_category_plot(category_name, filtered_transactions_df, budget_df, global_month_range, categories_data)
                    if category_fig:
                        st.plotly_chart(category_fig, use_container_width=True)
                    else:
                        st.info(f"No data available for {category_name}")

    # Raw Data Section
    st.header("📋 Raw Transaction Data")
    
    # Category filter for raw data
    st.subheader("Filter Options")
    
    # Create two columns for filters
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Add "All Categories" option (categories are already in API order)
        all_categories_option = ["All Categories"] + filtered_category_names
        selected_category = st.selectbox(
            "Select Category to Filter:",
            options=all_categories_option,
            index=0,
            help="Choose a specific category or 'All Categories' to see all transactions"
        )
    
    with filter_col2:
        # Split transaction filter
        if isinstance(filtered_transactions_df, pd.DataFrame) and 'is_subtransaction' in filtered_transactions_df.columns:
            split_filter_options = ['All Transactions', 'Regular Transactions Only', 'Split Transactions Only']
            selected_split_filter = st.selectbox(
                "Filter by Transaction Type:",
                options=split_filter_options,
                index=0,
                help="Choose to view all transactions, only regular transactions, or only split transactions"
            )
        else:
            selected_split_filter = 'All Transactions'
    
    # Filter data based on selections
    filtered_raw_data = filtered_transactions_df.copy()
    
    # Apply category filter
    if selected_category != "All Categories":
        filtered_raw_data = filtered_raw_data[filtered_raw_data['category'] == selected_category]
    
    # Apply split transaction filter
    if selected_split_filter == 'Regular Transactions Only':
        filtered_raw_data = filtered_raw_data[filtered_raw_data['is_subtransaction'] == False]
    elif selected_split_filter == 'Split Transactions Only':
        filtered_raw_data = filtered_raw_data[filtered_raw_data['is_subtransaction'] == True]
    
    # Ensure it's a DataFrame
    filtered_raw_data = pd.DataFrame(filtered_raw_data)
    
    # Display data info
    filter_info = f"Showing {len(filtered_raw_data)} transactions for: **{selected_category}**"
    if selected_split_filter != 'All Transactions':
        filter_info += f" ({selected_split_filter.lower()})"
    st.info(filter_info)
    
    # Display the raw data
    if isinstance(filtered_raw_data, pd.DataFrame) and not filtered_raw_data.empty:
        display_data = pd.DataFrame(filtered_raw_data)
        display_data['date'] = display_data['date'].apply(str)
        display_data['amount'] = display_data['amount'].round(2)
        column_order = ['date', 'category', 'category_group', 'amount', 'payee_name', 'memo', 'is_income']
        if isinstance(display_data, pd.DataFrame) and all(col in display_data.columns for col in column_order):
            display_data = display_data[column_order]
        display_data = pd.DataFrame(display_data)
        display_data = display_data.rename(columns={
            'date': 'Date',
            'category': 'Category',
            'category_group': 'Category Group',
            'amount': 'Amount (€)',
            'payee_name': 'Payee',
            'memo': 'Memo',
            'is_income': 'Is Income'
        })
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button for filtered data
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"ynab_transactions_{selected_category.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No transactions found for category: {selected_category}")

if __name__ == "__main__":
    main() 