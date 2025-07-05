import streamlit as st
import pandas as pd
from datetime import date
import os

from mynab.utils import (
    get_ynab_data,
    process_categories_data,
    process_transactions_data,
    process_months_data,
    filter_data_by_date_range,
    safe_strftime,
    get_default_date_range,
    get_excluded_groups,
    get_default_category_groups,
)

# Page configuration
st.set_page_config(page_title="YNAB Budget Dashboard", page_icon="💰", layout="wide")

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


def initialize_session_state():
    """Initialize session state variables"""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "start_date" not in st.session_state:
        st.session_state.start_date = None
    if "end_date" not in st.session_state:
        st.session_state.end_date = None
    if "selected_category_groups" not in st.session_state:
        st.session_state.selected_category_groups = []
    if "transactions_df" not in st.session_state:
        st.session_state.transactions_df = None
    if "budget_df" not in st.session_state:
        st.session_state.budget_df = None
    if "categories_data" not in st.session_state:
        st.session_state.categories_data = None
    if "category_groups" not in st.session_state:
        st.session_state.category_groups = None


def load_data():
    """Load and cache YNAB data"""
    if not st.session_state.data_loaded:
        # Get API token from environment
        api_token = os.getenv("YNAB_API_KEY")

        if not api_token:
            st.error("YNAB API key not found in environment variables.")
            st.info("""
            Please set your YNAB API key in the .env file:
            1. Edit the .env file in your project directory
            2. Replace 'your_ynab_api_token_here' with your actual API token
            3. Get your API token from YNAB Account Settings > Developer Settings
            """)
            return False

        # Fetch data
        with st.spinner("Fetching data from YNAB..."):
            (
                budget_id,
                budget_name,
                categories_response,
                transactions_response,
                months_response,
            ) = get_ynab_data(api_token)

        if not budget_id:
            return False

        # Process data
        categories_data, category_groups = process_categories_data(categories_response)
        transactions_df = process_transactions_data(
            transactions_response, categories_data
        )
        budget_df = process_months_data(months_response, categories_data)

        # Store in session state
        st.session_state.budget_name = budget_name
        st.session_state.transactions_df = transactions_df
        st.session_state.budget_df = budget_df
        st.session_state.categories_data = categories_data
        st.session_state.category_groups = category_groups
        st.session_state.data_loaded = True

        return True

    return True


def setup_sidebar():
    """Setup sidebar with filters"""
    with st.sidebar:
        st.header("📅 Date Range Filter")

        st.info(f"**{st.session_state.budget_name}**")

        # Get default date range
        default_start_date, default_end_date = get_default_date_range()

        # Get the actual date range from the data
        if (
            st.session_state.transactions_df is not None
            and not st.session_state.transactions_df.empty
        ):
            filtered_transactions_df = st.session_state.transactions_df.copy()
            filtered_transactions_df["date"] = pd.to_datetime(
                filtered_transactions_df["date"]
            )
            data_start_date = filtered_transactions_df["date"].min().date()
            data_end_date = filtered_transactions_df["date"].max().date()

            # Use data range as defaults if available, but cap end date to last day of prior month
            default_start_date = data_start_date
            default_end_date = min(data_end_date, default_end_date)

        # Date picker in sidebar
        today = date.today()
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.start_date or default_start_date,
            min_value=date(2010, 1, 1),
            max_value=today,
            help="Select the start date for filtering data",
        )

        end_date = st.date_input(
            "End Date",
            value=st.session_state.end_date or default_end_date,
            min_value=date(2010, 1, 1),
            max_value=today,
            help="Select the end date for filtering data",
        )

        # Validate date range
        if start_date > end_date:
            st.error("Start date must be before end date!")
            return False

        # Store in session state
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date

        # Category Group Filter in sidebar
        st.header("📊 Category Group Filter")

        if st.session_state.category_groups is not None:
            # Get all category group names (excluding the specified groups)
            excluded_groups = get_excluded_groups()
            category_group_names = sorted(
                [
                    group
                    for group in st.session_state.category_groups.keys()
                    if group not in excluded_groups
                ]
            )

            # Set default values for multiselect
            default_groups = get_default_category_groups()
            # Filter default groups to only include those that exist in the data
            available_defaults = [
                group for group in default_groups if group in category_group_names
            ]

            selected_category_groups = st.multiselect(
                "Select category groups to include:",
                options=category_group_names,
                default=st.session_state.selected_category_groups or available_defaults,
                help="Choose which category groups to include in the analysis. Leave empty to show all groups.",
            )

            # If no groups are selected, show all groups
            if not selected_category_groups:
                selected_category_groups = category_group_names

            # Store in session state
            st.session_state.selected_category_groups = selected_category_groups

    return True


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()

    # Load data
    if not load_data():
        return

    # Setup sidebar
    if not setup_sidebar():
        return

    # Navigation
    pg = st.navigation(
        [
            st.Page("pages/overview.py", title="Overview", icon="📊"),
            st.Page("pages/categories.py", title="Categories", icon="📋"),
        ]
    )
    pg.run()


if __name__ == "__main__":
    main()
