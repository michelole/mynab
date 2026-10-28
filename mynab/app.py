import os
from collections.abc import Mapping, Sequence
from datetime import date

import streamlit as st
from streamlit.runtime.pages_manager import PagesManager

from mynab.api import (
    get_ynab_data,
    process_categories_data,
    process_months_data,
    process_transactions_data,
)
from mynab.charts import inject_custom_css
from mynab.config import settings
from mynab.models import (
    DEFAULT_MONTH_RANGE_PRESET,
    DEFAULT_MOVING_AVERAGE_WINDOW,
    MONTH_RANGE_PRESET_OPTIONS,
    MOVING_AVERAGE_WINDOW_OPTIONS,
    CategoryInfo,
    get_default_categories,
    get_default_category_groups,
    get_excluded_groups,
)
from mynab.transforms import (
    clamp_month_range,
    dates_from_month_selection,
    format_month_option,
    get_default_month_range,
    list_available_months,
    month_start,
    resolve_month_range_preset,
)

PagesManager.uses_pages_directory = False


def initialize_session_state() -> None:
    """Initialize session state variables"""
    defaults: dict[str, object] = {
        "data_loaded": False,
        "start_date": None,
        "end_date": None,
        "selected_category_groups": [],
        "selected_categories": [],
        "transactions_df": None,
        "budget_df": None,
        "categories_data": None,
        "category_groups": None,
        "accounts_list": None,
        "global_scale_enabled": True,
        "moving_average_window": DEFAULT_MOVING_AVERAGE_WINDOW,
        "month_range_preset": DEFAULT_MONTH_RANGE_PRESET,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_api_token() -> str | None:
    """Retrieve YNAB API token from environment or config."""
    return os.getenv("YNAB_API_KEY") or settings.ynab_api_key


def _resolve_preset_dates(preset: object, available_months: Sequence[date]) -> tuple[date, date]:
    """Compute inclusive start and end dates for the given preset."""
    resolved = (
        preset
        if isinstance(preset, str) and preset in MONTH_RANGE_PRESET_OPTIONS
        else DEFAULT_MONTH_RANGE_PRESET
    )
    if resolved == "Custom":
        start_raw, end_raw = get_default_month_range()
        start_month, end_month = clamp_month_range(start_raw, end_raw, available_months)
    else:
        start_month, end_month = resolve_month_range_preset(resolved, available_months)
    return dates_from_month_selection(start_month, end_month)


def _apply_initial_filter_defaults() -> None:
    """Initialize default date range and category filters after loading data."""
    if st.session_state.get("start_date") is None or st.session_state.get("end_date") is None:
        available_months = list_available_months(st.session_state.transactions_df)
        start_date, end_date = _resolve_preset_dates(
            st.session_state.get("month_range_preset"), available_months
        )
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date

    category_groups = st.session_state.get("category_groups")
    if not isinstance(category_groups, Mapping):
        return

    if not st.session_state.get("selected_category_groups"):
        excluded = set(get_excluded_groups())
        all_groups = sorted(g for g in category_groups if g not in excluded)
        default_groups = [g for g in get_default_category_groups() if g in all_groups]
        st.session_state.selected_category_groups = default_groups or all_groups

    if not st.session_state.get("selected_categories"):
        available_cats = _get_categories_for_groups(
            category_groups, st.session_state.selected_category_groups
        )
        default_cats = [c for c in get_default_categories() if c in available_cats]
        st.session_state.selected_categories = default_cats or available_cats


def load_data() -> bool:
    """Load and cache YNAB data"""
    initialize_session_state()
    if st.session_state.data_loaded:
        return True

    api_token = _get_api_token()
    if api_token is None or api_token == "":
        st.error("YNAB API key not found in environment variables.")
        st.info("""
        Please set your YNAB API key in the .env file:
        1. Edit the .env file in your project directory
        2. Replace 'your_ynab_api_token_here' with your actual API token
        3. Get your API token from YNAB Account Settings > Developer Settings
        """)
        return False

    with st.spinner("Fetching data from YNAB..."):
        (
            budget_id,
            budget_name,
            categories_response,
            transactions_response,
            months_response,
            accounts_list,
        ) = get_ynab_data(api_token)

    if budget_id is None or categories_response is None or transactions_response is None:
        return False

    categories_data, category_groups = process_categories_data(categories_response)
    st.session_state.budget_id = budget_id
    st.session_state.budget_name = budget_name
    st.session_state.transactions_df = process_transactions_data(
        transactions_response, categories_data
    )
    st.session_state.budget_df = process_months_data(months_response, categories_data)
    st.session_state.categories_data = categories_data
    st.session_state.category_groups = category_groups
    st.session_state.accounts_list = accounts_list
    st.session_state.data_loaded = True
    _apply_initial_filter_defaults()
    return True


def ensure_data_loaded() -> bool:
    """Ensure session state and YNAB data are initialized and loaded."""
    return load_data()


def _resolve_custom_month_defaults(
    available_months: Sequence[date],
) -> tuple[int, int]:
    """Determine initial start and end months for custom month selection."""
    default_start_raw, default_end_raw = get_default_month_range()
    default_start, default_end = clamp_month_range(
        default_start_raw, default_end_raw, available_months
    )
    if st.session_state.start_date and st.session_state.end_date:
        current_start = month_start(st.session_state.start_date)
        current_end = month_start(st.session_state.end_date)
    else:
        current_start, current_end = default_start, default_end

    start_idx = (
        available_months.index(current_start)
        if current_start in available_months
        else available_months.index(default_start)
    )
    end_idx = (
        available_months.index(current_end)
        if current_end in available_months
        else available_months.index(default_end)
    )
    return start_idx, end_idx


def _render_custom_month_selectors(available_months: Sequence[date]) -> bool:
    """Render start and end month selectboxes for Custom range preset."""
    month_labels = [format_month_option(m) for m in available_months]
    label_to_month = dict(zip(month_labels, available_months, strict=True))
    start_idx, end_idx = _resolve_custom_month_defaults(available_months)

    start_label = st.selectbox(
        "Start month",
        month_labels,
        index=start_idx,
        help="First month included in charts and totals",
    )
    end_label = st.selectbox(
        "End month",
        month_labels,
        index=end_idx,
        help="Last month included in charts and totals",
    )

    start_month = label_to_month[start_label]
    end_month = label_to_month[end_label]
    if start_month > end_month:
        st.error("Start month must be before or equal to end month!")
        return False

    start_date, end_date = dates_from_month_selection(start_month, end_month)
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    return True


def _setup_month_range_section() -> bool:
    """Setup month range selection in sidebar"""
    st.header("📅 Month Range")

    available_months = list_available_months(st.session_state.transactions_df)
    preset_labels = MONTH_RANGE_PRESET_OPTIONS
    current_preset = st.session_state.month_range_preset
    if current_preset not in preset_labels:
        current_preset = DEFAULT_MONTH_RANGE_PRESET

    preset = st.selectbox(
        "Time range",
        preset_labels,
        index=preset_labels.index(current_preset),
        help="Quick range presets; choose Custom to pick start and end months",
    )
    st.session_state.month_range_preset = preset

    if preset != "Custom":
        start_month, end_month = resolve_month_range_preset(preset, available_months)
        start_date, end_date = dates_from_month_selection(start_month, end_month)
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        return True

    return _render_custom_month_selectors(available_months)


def _setup_plot_settings_section() -> None:
    """Setup plot settings in sidebar"""
    st.header("📏 Plot Settings")

    ma_labels = list(MOVING_AVERAGE_WINDOW_OPTIONS.keys())
    ma_label = next(
        (
            label
            for label, months in MOVING_AVERAGE_WINDOW_OPTIONS.items()
            if months == st.session_state.moving_average_window
        ),
        "12 months (weighted)",
    )
    selected_ma_label = st.selectbox(
        "Moving average window",
        options=ma_labels,
        index=ma_labels.index(ma_label),
        help=(
            "Lookback for the moving average line on plots. "
            "Does not affect the 12-month forecast trend."
        ),
    )
    st.session_state.moving_average_window = MOVING_AVERAGE_WINDOW_OPTIONS[selected_ma_label]

    st.session_state.global_scale_enabled = st.checkbox(
        "Global Scale",
        value=st.session_state.global_scale_enabled,
        help=(
            "When enabled, all plots in a section will share the same "
            "y-axis scale for better comparison."
        ),
    )


def _multiselect_with_fallback(
    label: str,
    options: list[str],
    current_selection: Sequence[str],
    fallback_defaults: Sequence[str],
    help_text: str,
) -> list[str]:
    """Render a multiselect widget that falls back to all options when cleared."""
    valid_current = [item for item in current_selection if item in options]
    valid_defaults = [item for item in fallback_defaults if item in options]
    selected = st.multiselect(
        label,
        options=options,
        default=valid_current or valid_defaults,
        help=help_text,
    )
    return selected or options


def _get_categories_for_groups(
    category_groups: Mapping[str, Sequence[CategoryInfo]],
    selected_groups: Sequence[str],
) -> list[str]:
    """Collect sorted category names belonging to the selected category groups."""
    names = [
        cat["name"] for group_name in selected_groups for cat in category_groups.get(group_name, [])
    ]
    return sorted(names)


def _setup_category_filters_section() -> None:
    """Setup category group and category filters in sidebar"""
    st.header("📊 Category Groups")

    category_groups = st.session_state.category_groups
    if category_groups is None:
        return

    excluded_groups = set(get_excluded_groups())
    category_group_names = sorted(
        group for group in category_groups if group not in excluded_groups
    )

    selected_category_groups = _multiselect_with_fallback(
        "Select category groups to include:",
        category_group_names,
        st.session_state.selected_category_groups,
        get_default_category_groups(),
        "Choose which category groups to include in the analysis. Leave empty to show all groups.",
    )
    st.session_state.selected_category_groups = selected_category_groups

    st.header("📋 Categories")
    available_categories = _get_categories_for_groups(category_groups, selected_category_groups)
    st.session_state.selected_categories = _multiselect_with_fallback(
        "Select categories to include:",
        available_categories,
        st.session_state.selected_categories,
        get_default_categories(),
        (
            "Choose which categories to include in the analysis. "
            "Leave empty to show all categories from selected groups."
        ),
    )


def setup_sidebar() -> bool:
    """Setup sidebar with filters"""
    with st.sidebar:
        if not _setup_month_range_section():
            return False
        _setup_plot_settings_section()
        _setup_category_filters_section()

    return True


def main() -> None:
    """Main application function"""
    st.set_page_config(page_title="YNAB Budget Dashboard", page_icon="💰", layout="wide")
    inject_custom_css()
    initialize_session_state()

    if not load_data():
        return

    if not setup_sidebar():
        return

    pg = st.navigation(
        {
            "": [
                st.Page("pages/overview.py", title="Overview", icon="📊"),
                st.Page("pages/categories.py", title="Categories", icon="📋"),
                st.Page("pages/accounts.py", title="Accounts", icon="🏦"),
                st.Page("pages/budgets.py", title="Budgets", icon="🪣"),
            ],
            "Raw Data": [
                st.Page("pages/transactions.py", title="Transactions", icon="📄"),
                st.Page("pages/budget_data.py", title="Plan", icon="💰"),
            ],
        }
    )
    pg.run()


if __name__ == "__main__":
    main()
