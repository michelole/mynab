import streamlit as st
import pandas as pd
from mynab.utils import (
    get_excluded_groups,
)

# Page configuration
st.set_page_config(
    page_title="Budget Data - YNAB Budget Dashboard", page_icon="💰", layout="wide"
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
        '<h1 class="main-header">💰 Budget Data</h1>',
        unsafe_allow_html=True,
    )

    # Check if data is loaded in session state
    if not st.session_state.get("data_loaded", False):
        st.error("Data not loaded. Please go to the main page first.")
        return

    # Get data from session state
    budget_df = st.session_state.budget_df
    selected_category_groups = st.session_state.selected_category_groups

    if budget_df is None:
        st.error("Data not available. Please go to the main page first.")
        return

    # Get selected categories from session state
    selected_categories = st.session_state.get("selected_categories", [])

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

    if budget_df is not None and not budget_df.empty:
        st.dataframe(budget_df, use_container_width=True, height=600)
    else:
        st.info("No budget data available - using transaction data for analysis")


if __name__ == "__main__":
    main()
