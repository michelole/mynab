import streamlit as st

from mynab.utils import (
    filter_budget_by_selection,
    render_page_header,
)


def main():
    render_page_header("💰 Budget Data")

    if not st.session_state.get("data_loaded", False):
        st.error("Data not loaded. Please go to the main page first.")
        return

    budget_df = st.session_state.budget_df
    if budget_df is None:
        st.error("Data not available. Please go to the main page first.")
        return

    selected_category_groups = st.session_state.selected_category_groups
    selected_categories = st.session_state.get("selected_categories", [])
    budget_df = filter_budget_by_selection(
        budget_df, selected_category_groups, selected_categories
    )

    if budget_df is not None and not budget_df.empty:
        st.dataframe(budget_df, width="stretch", height=600)
    else:
        st.info("No budget data available - using transaction data for analysis")


if __name__ == "__main__":
    main()
