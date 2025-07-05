import streamlit as st
import pandas as pd
from datetime import datetime
from mynab.utils import (
    filter_data_by_date_range,
    get_excluded_groups,
)

# Page configuration
st.set_page_config(
    page_title="Transactions - YNAB Budget Dashboard", page_icon="📄", layout="wide"
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


def main():
    st.markdown(
        '<h1 class="main-header">📄 Transaction Data</h1>',
        unsafe_allow_html=True,
    )

    # Check if data is loaded in session state
    if not st.session_state.get("data_loaded", False):
        st.error("Data not loaded. Please go to the main page first.")
        return

    # Get data from session state
    transactions_df = st.session_state.transactions_df
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    selected_category_groups = st.session_state.selected_category_groups

    if transactions_df is None:
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

    # Filter transactions to only include selected category groups and categories
    if (
        isinstance(filtered_transactions_df, pd.DataFrame)
        and not filtered_transactions_df.empty
    ):
        filtered_transactions_df = filtered_transactions_df[
            (filtered_transactions_df["category_group"].isin(selected_category_groups))
            & (filtered_transactions_df["category"].isin(selected_categories))
        ].copy()

    # Display filtered transactions
    st.dataframe(filtered_transactions_df, use_container_width=True)


if __name__ == "__main__":
    main()
