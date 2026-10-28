import streamlit as st

from mynab.app import ensure_data_loaded
from mynab.charts import render_page_header
from mynab.transforms import filter_budget_by_selection


def main() -> None:
    render_page_header("💰 Plan")

    if not ensure_data_loaded():
        return

    budget_df = st.session_state.budget_df
    if budget_df is None:
        st.error("Data not available. Please go to the main page first.")
        return

    selected_category_groups = st.session_state.selected_category_groups
    raw_categories = st.session_state.get("selected_categories")
    selected_categories = (
        [str(c) for c in raw_categories] if isinstance(raw_categories, list) else []
    )
    budget_df = filter_budget_by_selection(budget_df, selected_category_groups, selected_categories)

    if not budget_df.empty:
        st.dataframe(budget_df, width="stretch", height=600)
    else:
        st.info("No budget data available - using transaction data for analysis")


if __name__ == "__main__":
    main()
