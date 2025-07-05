import streamlit as st
from ynab.configuration import Configuration
from ynab.api_client import ApiClient
from ynab.api.budgets_api import BudgetsApi
from ynab.api.categories_api import CategoriesApi
from ynab.api.transactions_api import TransactionsApi
from ynab.api.months_api import MonthsApi
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ynab_data(access_token):
    """Fetch data from YNAB API"""
    try:
        configuration = Configuration(access_token=access_token)
        with ApiClient(configuration) as api_client:
            # Get budgets
            budgets_api = BudgetsApi(api_client)
            budgets_response = budgets_api.get_budgets()
            
            if not budgets_response.data.budgets:
                st.error("No budgets found in your YNAB account.")
                return None, None, None, None, None
            
            budget_id = budgets_response.data.budgets[0].id
            budget_name = budgets_response.data.budgets[0].name
            
            # Get categories
            categories_api = CategoriesApi(api_client)
            categories_response = categories_api.get_categories(budget_id)
            
            # Get transactions for the last 24 months
            transactions_api = TransactionsApi(api_client)
            
            # Calculate date range (last 24 months)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            transactions_response = transactions_api.get_transactions(
                budget_id,
                since_date=start_date.date()
            )
            
            # Get current month budget data for categories
            months_api = MonthsApi(api_client)
            current_month = datetime.now().replace(day=1).date()  # Use first day of current month
            months_response = months_api.get_budget_month(budget_id, current_month)
            
            return budget_id, budget_name, categories_response, transactions_response, months_response
            
    except Exception as e:
        st.error(f"Error connecting to YNAB API: {str(e)}")
        return None, None, None, None, None

def process_categories_data(categories_response):
    """Process categories data into a structured format, excluding specified groups"""
    categories_data = []
    category_groups = {}
    
    # Categories to exclude
    excluded_groups = ['Internal Master Category', 'Uncategorized', 'Credit Card Payments', 'Hidden Categories']
    
    for group in categories_response.data.category_groups:
        group_name = group.name
        
        # Skip excluded category groups
        if group_name in excluded_groups:
            continue
            
        category_groups[group_name] = []
        
        for category in group.categories:
            if not category.hidden and not category.deleted:
                # Extract target/goal information
                target_amount = None
                target_type = None
                target_date = None
                
                if hasattr(category, 'goal_target') and category.goal_target:
                    target_amount = category.goal_target / 1000  # Convert from millidollars
                
                if hasattr(category, 'goal_type') and category.goal_type:
                    target_type = category.goal_type
                
                if hasattr(category, 'goal_target_month') and category.goal_target_month:
                    target_date = category.goal_target_month
                
                category_data = {
                    'id': category.id,
                    'name': category.name,
                    'group': group_name,
                    'category_group_id': group.id,
                    'target_amount': target_amount,
                    'target_type': target_type,
                    'target_date': target_date
                }
                categories_data.append(category_data)
                category_groups[group_name].append(category_data)
    
    return categories_data, category_groups

def process_transactions_data(transactions_response, categories_data):
    """Process transactions data into a structured format, handling split transactions"""
    transactions = []
    
    for transaction in transactions_response.data.transactions:
        # Skip zero amount transactions
        if transaction.amount == 0:
            continue
            
        # Get the date
        transaction_date = None
        if hasattr(transaction, 'var_date'):
            transaction_date = transaction.var_date
        elif hasattr(transaction, 'date'):
            transaction_date = transaction.date
        else:
            # Skip transactions without a date
            continue
        
        # Check if this transaction has subtransactions (split transaction)
        has_subtransactions = hasattr(transaction, 'subtransactions') and transaction.subtransactions
        
        if has_subtransactions:
            # Process each subtransaction
            for subtransaction in transaction.subtransactions:
                if subtransaction.amount == 0:
                    continue
                    
                # Find category name and group for subtransaction
                category_name = subtransaction.category_name or ""
                category_group = ""
                
                if subtransaction.category_id:
                    for cat in categories_data:
                        if cat['id'] == subtransaction.category_id:
                            category_group = cat['group']
                            break
                
                # Determine if this is income or expense
                is_income = subtransaction.amount > 0
                
                transactions.append({
                    'date': transaction_date,
                    'amount': subtransaction.amount / 1000,  # Convert from millidollars
                    'category': category_name,
                    'category_group': category_group,
                    'payee_name': transaction.payee_name or "",
                    'memo': subtransaction.memo or transaction.memo or "",
                    'is_income': is_income,
                    'transaction_id': transaction.id,
                    'is_subtransaction': True
                })
        else:
            # Process regular transaction (not split)
            category_name = transaction.category_name or ""
            category_group = ""
            
            # Find category group from categories_data
            if transaction.category_id:
                for cat in categories_data:
                    if cat['id'] == transaction.category_id:
                        category_group = cat['group']
                        break
            
            # Determine if this is income or expense
            is_income = transaction.amount > 0
            
            transactions.append({
                'date': transaction_date,
                'amount': transaction.amount / 1000,  # Convert from millidollars
                'category': category_name,
                'category_group': category_group,
                'payee_name': transaction.payee_name or "",
                'memo': transaction.memo or "",
                'is_income': is_income,
                'transaction_id': transaction.id,
                'is_subtransaction': False
            })
    
    df = pd.DataFrame(transactions)
    
    # Convert date column to datetime if it exists
    if not df.empty and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def process_months_data(months_response, categories_data):
    """Process months data for budget information"""
    budget_data = []
    
    if months_response and months_response.data.month:
        month_data = months_response.data.month
        
        # Extract category budgets from the current month
        if hasattr(month_data, 'categories'):
            for category in month_data.categories:
                if category.budgeted != 0 or category.activity != 0:
                    # Find category name and target information
                    category_name = "Uncategorized"
                    category_group = "Uncategorized"
                    target_amount = None
                    target_type = None
                    target_date = None
                    
                    for cat in categories_data:
                        if cat['id'] == category.id:
                            category_name = cat['name']
                            category_group = cat['group']
                            target_amount = cat.get('target_amount')
                            target_type = cat.get('target_type')
                            target_date = cat.get('target_date')
                            break
                    
                    budget_data.append({
                        'category': category_name,
                        'category_group': category_group,
                        'budgeted': category.budgeted / 1000,  # Convert from millidollars
                        'activity': category.activity / 1000,  # Convert from millidollars
                        'available': category.balance / 1000,   # Convert from millidollars
                        'target_amount': target_amount,
                        'target_type': target_type,
                        'target_date': target_date
                    })
    
    df = pd.DataFrame(budget_data)
    return df

def calculate_moving_average(data, window=12):
    """Calculate moving average for the given data"""
    if len(data) < 2:
        return data
    
    # If we have less than the window size, use all available data
    actual_window = min(window, len(data))
    return data.rolling(window=actual_window, min_periods=1).mean()

def calculate_forecast_trend(data, periods=3):
    """Calculate forecast trend using linear regression"""
    if len(data) < 2:
        return data
    
    x = np.arange(len(data))
    y = data.values
    
    # Simple linear regression
    coeffs = np.polyfit(x, y, 1)
    trend_line = np.polyval(coeffs, x)
    
    # Extend trend line for forecast
    future_x = np.arange(len(data), len(data) + periods)
    forecast = np.polyval(coeffs, future_x)
    
    return pd.Series(trend_line, index=data.index), pd.Series(forecast, index=future_x)

def calculate_category_group_averages(group_name, transactions_df, months=3):
    """Calculate average spending for a category group over the last N months"""
    # Filter data for this category group
    group_transactions = transactions_df[transactions_df['category_group'] == group_name].copy()
    
    if group_transactions.empty:
        return 0
    
    # Convert date and group by month
    group_transactions['date'] = pd.to_datetime(group_transactions['date'])
    group_transactions['month'] = group_transactions['date'].dt.to_period('M')
    
    # Get monthly totals
    monthly_expenses = group_transactions.groupby('month')['amount'].sum()
    
    if monthly_expenses.empty:
        return 0
    
    # Sort by month and get the last N months
    monthly_expenses = monthly_expenses.sort_index()
    last_n_months = monthly_expenses.tail(months)
    
    # Calculate average (convert to positive for display)
    avg_amount = abs(last_n_months.mean())
    
    return avg_amount

def calculate_category_group_available_budget(group_name, budget_df):
    """Calculate total available budget for a category group"""
    if budget_df.empty:
        return 0
    
    # Filter budget data for this category group
    group_budget = budget_df[budget_df['category_group'] == group_name]
    
    if group_budget.empty:
        return 0
    
    # Sum the available amounts for all categories in this group
    total_available = group_budget['available'].sum()
    
    return total_available

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

def filter_data_by_date_range(transactions_df, start_date, end_date):
    """Filter transactions dataframe by date range"""
    if transactions_df.empty:
        return transactions_df
    
    # Ensure date column is datetime
    filtered_df = transactions_df.copy()
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    
    # Filter by date range
    mask = (filtered_df['date'] >= pd.Timestamp(start_date)) & (filtered_df['date'] <= pd.Timestamp(end_date))
    return filtered_df[mask]

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
    excluded_groups = ['Internal Master Category', 'Uncategorized', 'Credit Card Payments', 'Hidden Categories']
    filtered_transactions_df = transactions_df[~transactions_df['category_group'].isin(excluded_groups)].copy()
    
    # Sidebar date picker
    st.sidebar.header("📅 Date Range Filter")
    
    # Calculate default date range (last 12 months)
    today = date.today()
    default_start_date = today - timedelta(days=365)
    default_end_date = today
    
    # Get the actual date range from the data
    if isinstance(filtered_transactions_df, pd.DataFrame) and not filtered_transactions_df.empty:
        filtered_transactions_df['date'] = pd.to_datetime(filtered_transactions_df['date'])
        data_start_date = filtered_transactions_df['date'].min().date()
        data_end_date = filtered_transactions_df['date'].max().date()
        
        # Use data range as defaults if available
        default_start_date = data_start_date
        default_end_date = data_end_date
    
    # Date picker in sidebar
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
    
    # Show date range info in sidebar
    st.sidebar.success(f"📊 Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    st.sidebar.info(f"📈 {len(filtered_transactions_df)} transactions in selected range")
    
    # Calculate global month range from filtered data
    if isinstance(filtered_transactions_df, pd.DataFrame) and not filtered_transactions_df.empty:
        all_transactions = filtered_transactions_df.copy()
        all_transactions['date'] = pd.to_datetime(all_transactions['date'])
        all_transactions = all_transactions.reset_index(drop=True)
        all_transactions['month'] = all_transactions['date'].dt.to_period('M')
        
        # Get the earliest and latest months from filtered data
        earliest_month = all_transactions['month'].min()
        latest_month = all_transactions['month'].max()
        
        # Convert to datetime for date_range
        earliest_date = earliest_month.to_timestamp()
        latest_date = latest_month.to_timestamp()
        
        # Create global month range using month start frequency to ensure all months are included
        global_month_range = pd.date_range(start=earliest_date, end=latest_date, freq='MS')
    else:
        # If no data in range, create a default range
        global_month_range = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq='MS')
        earliest_date = pd.Timestamp(start_date)
        latest_date = pd.Timestamp(end_date)
    
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
    
    # Get all category group names (excluding the specified groups)
    category_group_names = sorted([group for group in category_groups.keys() if group not in excluded_groups])
    
    # Set default values for multiselect
    default_groups = ["Lazer", "Necessidades"]
    # Filter default groups to only include those that exist in the data
    available_defaults = [group for group in default_groups if group in category_group_names]
    
    selected_category_groups = st.multiselect(
        "Select category groups to display:",
        options=category_group_names,
        default=available_defaults,
        help="Choose which category groups to include in the analysis. Leave empty to show all groups."
    )
    
    # If no groups are selected, show all groups
    if not selected_category_groups:
        selected_category_groups = category_group_names
    
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
                target_summary = categories_with_targets[['name', 'group', 'target_amount', 'target_type', 'target_date']].copy()
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