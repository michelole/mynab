import streamlit as st
import ynab
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ynab_data(access_token):
    """Fetch data from YNAB API"""
    try:
        configuration = ynab.Configuration(access_token=access_token)
        with ynab.ApiClient(configuration) as api_client:
            # Get budgets
            budgets_api = ynab.BudgetsApi(api_client)
            budgets_response = budgets_api.get_budgets()
            
            if not budgets_response.data.budgets:
                st.error("No budgets found in your YNAB account.")
                return None, None, None, None, None
            
            budget_id = budgets_response.data.budgets[0].id
            budget_name = budgets_response.data.budgets[0].name
            
            # Get categories
            categories_api = ynab.CategoriesApi(api_client)
            categories_response = categories_api.get_categories(budget_id)
            
            # Get transactions for the last 24 months
            transactions_api = ynab.TransactionsApi(api_client)
            
            # Calculate date range (last 24 months)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            transactions_response = transactions_api.get_transactions(
                budget_id,
                since_date=start_date.strftime('%Y-%m-%d')
            )
            
            # Get current month budget data for categories
            months_api = ynab.MonthsApi(api_client)
            current_month = datetime.now().strftime('%Y-%m-01')  # Use first day of current month
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
    excluded_groups = ['Internal Master Category', 'Uncategorized', 'Credit Card Payments']
    
    for group in categories_response.data.category_groups:
        group_name = group.name
        
        # Skip excluded category groups
        if group_name in excluded_groups:
            continue
            
        category_groups[group_name] = []
        
        for category in group.categories:
            if not category.hidden and not category.deleted:
                category_data = {
                    'id': category.id,
                    'name': category.name,
                    'group': group_name,
                    'category_group_id': group.id
                }
                categories_data.append(category_data)
                category_groups[group_name].append(category_data)
    
    return categories_data, category_groups

def process_transactions_data(transactions_response, categories_data):
    """Process transactions data into a structured format"""
    transactions = []
    
    for transaction in transactions_response.data.transactions:
        # Include both income (positive) and expenses (negative)
        if transaction.amount != 0:  # Skip zero amounts
            # Find category name
            category_name = transaction.category_name or ""
            category_group = ""
            
            # Find category group from categories_data
            if transaction.category_id:
                for cat in categories_data:
                    if cat['id'] == transaction.category_id:
                        category_group = cat['group']
                        break
            
            # Get the date - use var_date as shown in debug output
            transaction_date = None
            if hasattr(transaction, 'var_date'):
                transaction_date = transaction.var_date
            elif hasattr(transaction, 'date'):
                transaction_date = transaction.date
            else:
                # Skip transactions without a date
                continue
            
            # Determine if this is income or expense
            is_income = transaction.amount > 0
            
            transactions.append({
                'date': transaction_date,
                'amount': -transaction.amount / 1000,  # Convert from millidollars and flip sign
                'category': category_name,
                'category_group': category_group,
                'payee_name': transaction.payee_name or "",
                'memo': transaction.memo or "",
                'is_income': is_income
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
                    # Find category name
                    category_name = "Uncategorized"
                    category_group = "Uncategorized"
                    
                    for cat in categories_data:
                        if cat['id'] == category.id:
                            category_name = cat['name']
                            category_group = cat['group']
                            break
                    
                    budget_data.append({
                        'category': category_name,
                        'category_group': category_group,
                        'budgeted': category.budgeted / 1000,  # Convert from millidollars
                        'activity': category.activity / 1000,  # Convert from millidollars
                        'available': category.balance / 1000   # Convert from millidollars
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

def create_category_plot(category_name, transactions_df, budget_df, global_month_range):
    """Create comprehensive plot for a single category"""
    # Filter data for this category
    cat_transactions = transactions_df[transactions_df['category'] == category_name].copy()
    
    # Get budget data for this category
    cat_budget = pd.DataFrame()
    if not budget_df.empty and 'category' in budget_df.columns:
        cat_budget = budget_df[budget_df['category'] == category_name].copy()
    
    if cat_transactions.empty and cat_budget.empty:
        return None
    
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
        
        # Bar chart for actual expenses (including 0 values)
        fig.add_trace(
            go.Bar(
                x=complete_monthly_data['month'],
                y=complete_monthly_data['amount'],
                name='Actual Expenses',
                marker_color='#ff7f0e',
                opacity=0.8
            )
        )
        
        # Moving average line (if we have enough data)
        if len(complete_monthly_data) >= 3:
            moving_avg = calculate_moving_average(complete_monthly_data['amount'])
            
            fig.add_trace(
                go.Scatter(
                    x=complete_monthly_data['month'],
                    y=moving_avg,
                    name='12-Month Moving Average',
                    line=dict(color='#1f77b4', width=2, dash='dash'),
                    mode='lines'
                )
            )
            
            # Forecast trend line
            trend_line, forecast = calculate_forecast_trend(complete_monthly_data['amount'])
            
            fig.add_trace(
                go.Scatter(
                    x=complete_monthly_data['month'],
                    y=trend_line,
                    name='12-Month Forecast Trend',
                    line=dict(color='#d62728', width=2),
                    mode='lines'
                )
            )
            
            # Add forecast extension
            future_months = pd.date_range(
                start=complete_monthly_data['month_date'].iloc[-1] + pd.DateOffset(months=1),
                periods=3,
                freq='ME'
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_months.strftime('%Y-%m'),
                    y=forecast,
                    name='Forecast (Next 3 Months)',
                    line=dict(color='#d62728', width=2, dash='dot'),
                    mode='lines'
                )
            )
        
        # Add budget line if available
        if not cat_budget.empty and 'budgeted' in cat_budget.columns:
            budget_amount = cat_budget['budgeted'].iloc[0]
            if budget_amount > 0:
                # Create a horizontal line across all months
                all_months = complete_monthly_data['month'].tolist()
                if len(complete_monthly_data) >= 3 and 'future_months' in locals():
                    # Include future months in the budget line
                    future_month_strs = future_months.strftime('%Y-%m').tolist()
                    all_months.extend(future_month_strs)
                
                fig.add_trace(
                    go.Scatter(
                        x=all_months,
                        y=[budget_amount] * len(all_months),
                        name=f'Monthly Budget (€{budget_amount:,.0f})',
                        line=dict(color='#2ca02c', width=3),
                        mode='lines'
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=f'{category_name} - Comprehensive Analysis',
        height=400,
        showlegend=True,
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
    excluded_groups = ['Internal Master Category', 'Uncategorized', 'Credit Card Payments']
    filtered_transactions_df = transactions_df[~transactions_df['category_group'].isin(excluded_groups)].copy()
    
    # Get all category names (excluding the specified groups)
    category_names = sorted([cat['name'] for cat in categories_data])
    
    # Calculate global month range from all categories
    all_transactions = filtered_transactions_df.copy()
    all_transactions['date'] = pd.to_datetime(all_transactions['date'])
    all_transactions['month'] = all_transactions['date'].dt.to_period('M')
    
    # Get the earliest and latest months from all data
    earliest_month = all_transactions['month'].min()
    latest_month = all_transactions['month'].max()
    
    # Convert to datetime for date_range
    earliest_date = earliest_month.to_timestamp()
    latest_date = latest_month.to_timestamp()
    
    # Include current month if it's not already in the data
    current_month = pd.Timestamp.now().replace(day=1)
    if current_month > latest_date:
        latest_date = current_month
    
    # Create global month range
    global_month_range = pd.date_range(start=earliest_date, end=latest_date, freq='ME')
    
    st.info(f"Displaying analysis for {len(category_names)} categories (excluding Internal Master Category, Uncategorized, and Credit Card Payments)")
    st.info(f"Date range: {earliest_date.strftime('%Y-%m')} to {latest_date.strftime('%Y-%m')}")
    
    # Display all category plots in a grid
    st.header("📊 All Category Plots")
    
    # Create a grid layout for the plots
    cols_per_row = 2
    for i in range(0, len(category_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(category_names):
                category_name = category_names[i + j]
                with col:
                    st.subheader(category_name)
                    category_fig = create_category_plot(category_name, filtered_transactions_df, budget_df, global_month_range)
                    if category_fig:
                        st.plotly_chart(category_fig, use_container_width=True)
                    else:
                        st.info(f"No data available for {category_name}")

    # Raw Data Section
    st.header("📋 Raw Transaction Data")
    
    # Category filter for raw data
    st.subheader("Filter Options")
    
    # Add "All Categories" option
    all_categories_option = ["All Categories"] + category_names
    selected_category = st.selectbox(
        "Select Category to Filter:",
        options=all_categories_option,
        index=0,
        help="Choose a specific category or 'All Categories' to see all transactions"
    )
    
    # Filter data based on selection
    if selected_category == "All Categories":
        filtered_raw_data = filtered_transactions_df.copy()
    else:
        filtered_raw_data = filtered_transactions_df[filtered_transactions_df['category'] == selected_category].copy()
    
    # Ensure it's a DataFrame
    filtered_raw_data = pd.DataFrame(filtered_raw_data)
    
    # Display data info
    st.info(f"Showing {len(filtered_raw_data)} transactions for: **{selected_category}**")
    
    # Display the raw data
    if not filtered_raw_data.empty:
        # Format the data for display
        display_data = filtered_raw_data.copy()
        display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
        display_data['amount'] = display_data['amount'].round(2)
        
        # Reorder columns for better display
        column_order = ['date', 'category', 'category_group', 'amount', 'payee_name', 'memo', 'is_income']
        display_data = display_data[column_order]
        
        # Rename columns for better display
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