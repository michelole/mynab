import streamlit as st
from mynab.utils import (
    filter_transactions_by_selection,
    render_page_header,
)

DISPLAY_COLUMNS = [
    "date",
    "import_payee_name_original",
    "payee_name",
    "amount",
    "category",
]


def main():
    render_page_header("📄 Transaction Data")

    if not st.session_state.get("data_loaded", False):
        st.error("Data not loaded. Please go to the main page first.")
        return

    transactions_df = st.session_state.transactions_df
    if transactions_df is None:
        st.error("Data not available. Please go to the main page first.")
        return

    filtered_transactions_df = filter_transactions_by_selection(
        transactions_df,
        st.session_state.selected_category_groups,
        st.session_state.get("selected_categories", []),
        start_date=st.session_state.start_date,
        end_date=st.session_state.end_date,
    )

    display_df = (
        filtered_transactions_df[DISPLAY_COLUMNS]
        if not filtered_transactions_df.empty
        else filtered_transactions_df
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "import_payee_name_original": st.column_config.TextColumn(
                "import_payee_name_original", width="large"
            ),
        },
    )


if __name__ == "__main__":
    main()
