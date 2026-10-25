from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from ynab.api import AccountsApi, CategoriesApi, MonthsApi, PlansApi, TransactionsApi
from ynab.api_client import ApiClient
from ynab.configuration import Configuration
from ynab.exceptions import OpenApiException

import config

# Load environment variables
load_dotenv()

DASHBOARD_CSS = """
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
"""


def inject_custom_css():
    """Inject shared dashboard CSS styles into the current Streamlit page."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def render_page_header(title):
    """Render shared CSS and styled main header for a page."""
    inject_custom_css()
    st.markdown(f'<h1 class="main-header">{title}</h1>', unsafe_allow_html=True)


def render_in_grid(items, render_item_fn, cols_per_row=2):
    """Render a list of items across a multi-column Streamlit grid."""
    for i in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, items[i : i + cols_per_row]):
            with col:
                render_item_fn(item)


def format_currency(amount):
    """Format currency amount with no cents and period as thousand separator"""
    if pd.isna(amount) or amount is None:
        return "€0"

    rounded_amount = round(amount)
    if rounded_amount >= 0:
        return f"€{rounded_amount:,}".replace(",", ".")
    return f"-€{abs(rounded_amount):,}".replace(",", ".")


def _extract_accounts_list(accounts_response):
    """Convert YNAB accounts response into a list of dictionaries."""
    accounts = getattr(accounts_response.data, "accounts", None) or []
    result = []
    for acc in accounts:
        acc_type = getattr(acc, "type", None)
        result.append(
            {
                "id": str(acc.id),
                "name": acc.name,
                "type": acc_type.value if acc_type is not None else None,
                "on_budget": getattr(acc, "on_budget", None),
                "closed": getattr(acc, "closed", None),
            }
        )
    return result


@st.cache_data(ttl=config.CACHE_TTL)
def get_ynab_data(access_token):
    """Fetch data from YNAB API"""
    try:
        configuration = Configuration(access_token=access_token)
        with ApiClient(configuration) as api_client:
            plans_response = PlansApi(api_client).get_plans()
            if not plans_response.data.plans:
                st.error("No plans found in your YNAB account.")
                return None, None, None, None, None, None

            plan = plans_response.data.plans[0]
            budget_id = str(plan.id)
            budget_name = plan.name

            categories_response = CategoriesApi(api_client).get_categories(budget_id)

            start_date = datetime.now() - timedelta(days=730)
            transactions_response = TransactionsApi(api_client).get_transactions(
                budget_id, since_date=start_date.date()
            )

            current_month = datetime.now().replace(day=1).date()
            months_response = MonthsApi(api_client).get_plan_month(
                budget_id, current_month
            )

            accounts_response = AccountsApi(api_client).get_accounts(budget_id)
            accounts_list = _extract_accounts_list(accounts_response)

            return (
                budget_id,
                budget_name,
                categories_response,
                transactions_response,
                months_response,
                accounts_list,
            )

    except (OpenApiException, OSError, ValueError, TypeError) as e:
        st.error(f"Error connecting to YNAB API: {e!s}")
        return None, None, None, None, None, None


def _build_category_dict(category, group_name, group_id):
    """Extract a structured dictionary from a YNAB category model."""
    goal_target = getattr(category, "goal_target", None)
    target_date = (
        getattr(category, "goal_target_date", None)
        or getattr(category, "goal_target_month", None)
        or None
    )
    return {
        "id": category.id,
        "name": category.name,
        "group": group_name,
        "category_group_id": group_id,
        "target_amount": (goal_target / 1000) if goal_target else None,
        "target_type": getattr(category, "goal_type", None) or None,
        "target_date": target_date,
    }


def process_categories_data(categories_response):
    """Process categories data into a structured format, excluding specified groups"""
    categories_data = []
    category_groups = {}
    excluded_groups = set(get_excluded_groups())

    for group in categories_response.data.category_groups:
        if group.name in excluded_groups:
            continue

        group_categories = [
            _build_category_dict(cat, group.name, group.id)
            for cat in group.categories
            if not cat.hidden and not cat.deleted
        ]
        category_groups[group.name] = group_categories
        categories_data.extend(group_categories)

    return categories_data, category_groups


def _index_categories_by_id(categories_data):
    """Build a lookup dict mapping category ID to category metadata."""
    return {cat["id"]: cat for cat in (categories_data or []) if "id" in cat}


def _find_category_group(category_id, categories_data):
    """Find category group name for a given category ID"""
    if not category_id:
        return ""
    if isinstance(categories_data, dict):
        cat = categories_data.get(category_id)
        return cat["group"] if cat else ""
    for cat in categories_data:
        if cat["id"] == category_id:
            return cat["group"]
    return ""


def _get_transaction_date(transaction):
    """Extract transaction date from YNAB transaction object"""
    return getattr(transaction, "var_date", None) or getattr(transaction, "date", None)


def _build_tx_record(item, parent_tx, transaction_date, cat_by_id):
    """Build a dictionary record for a transaction or split subtransaction."""
    category_id = getattr(item, "category_id", None)
    cat_info = cat_by_id.get(category_id) if category_id else None
    category_group = cat_info["group"] if cat_info else ""
    memo = getattr(item, "memo", None) or getattr(parent_tx, "memo", None) or ""
    return {
        "date": transaction_date,
        "amount": item.amount / 1000,
        "category": getattr(item, "category_name", None) or "",
        "category_group": category_group,
        "payee_name": getattr(parent_tx, "payee_name", None) or "",
        "memo": memo,
        "transaction_id": parent_tx.id,
        "account_id": getattr(parent_tx, "account_id", None),
        "account_name": getattr(parent_tx, "account_name", None),
        "import_payee_name": getattr(
            item,
            "import_payee_name",
            getattr(parent_tx, "import_payee_name", ""),
        ),
        "import_payee_name_original": getattr(
            item,
            "import_payee_name_original",
            getattr(parent_tx, "import_payee_name_original", ""),
        ),
    }


def _extract_transaction_records(transaction, cat_by_id):
    """Extract one or more flat records from a YNAB transaction (including splits)."""
    if transaction.amount == 0:
        return []
    transaction_date = _get_transaction_date(transaction)
    if transaction_date is None:
        return []

    subtransactions = getattr(transaction, "subtransactions", None)
    if subtransactions:
        return [
            _build_tx_record(sub, transaction, transaction_date, cat_by_id)
            for sub in subtransactions
            if sub.amount != 0
        ]
    return [_build_tx_record(transaction, transaction, transaction_date, cat_by_id)]


def process_transactions_data(transactions_response, categories_data):
    """Process transactions data into a structured format, handling split transactions"""
    cat_by_id = _index_categories_by_id(categories_data)
    transactions = []
    for transaction in transactions_response.data.transactions:
        transactions.extend(_extract_transaction_records(transaction, cat_by_id))

    df = pd.DataFrame(transactions)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _build_month_category_record(category, cat_by_id):
    """Build a budget DataFrame record from a month category object."""
    cat_info = cat_by_id.get(category.id, {})
    return {
        "category": cat_info.get("name", "Uncategorized"),
        "category_group": cat_info.get("group", "Uncategorized"),
        "budgeted": category.budgeted / 1000,
        "activity": category.activity / 1000,
        "available": category.balance / 1000,
        "target_amount": cat_info.get("target_amount"),
        "target_type": cat_info.get("target_type"),
        "target_date": cat_info.get("target_date"),
    }


def process_months_data(months_response, categories_data):
    """Process months data for budget information"""
    month_data = getattr(getattr(months_response, "data", None), "month", None)
    categories = getattr(month_data, "categories", None) or []
    cat_by_id = _index_categories_by_id(categories_data)

    budget_data = [
        _build_month_category_record(cat, cat_by_id)
        for cat in categories
        if cat.budgeted != 0 or cat.activity != 0
    ]
    return pd.DataFrame(budget_data)


MOVING_AVERAGE_WINDOW_OPTIONS = {
    "12 months (weighted)": "weighted",
    "3 months": 3,
    "6 months": 6,
    "12 months": 12,
    "24 months": 24,
}
DEFAULT_MOVING_AVERAGE_WINDOW = "weighted"
AVERAGE_WINDOWS = (3, 6, 12, 24)


def get_moving_average_window():
    """Moving average lookback from sidebar (plots only; forecast trend unchanged)."""
    window = st.session_state.get(
        "moving_average_window", DEFAULT_MOVING_AVERAGE_WINDOW
    )
    if window not in MOVING_AVERAGE_WINDOW_OPTIONS.values():
        return DEFAULT_MOVING_AVERAGE_WINDOW
    return window


def moving_average_label(months):
    if months == "weighted":
        return "12-Month Weighted Moving Average"
    return f"{months}-Month Moving Average"


def map_moving_average_to_months(ma_series, months_to_plot):
    ma_map = {str(p): v for p, v in zip(ma_series.index, ma_series.values)}
    return [ma_map.get(m, None) for m in months_to_plot]


def _group_monthly_amounts(df):
    """Group a transactions DataFrame by month period and sum amounts."""
    tx = df.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx["month"] = tx["date"].dt.to_period("M")
    return tx.groupby("month")["amount"].sum().sort_index()


def build_account_net_worth_series(account_name, transactions_df=None, end_period=None):
    """Cumulative net worth by month from full account history."""
    if transactions_df is None:
        transactions_df = st.session_state.get("transactions_df")
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    txf = transactions_df[transactions_df["account_name"] == account_name]
    if txf.empty:
        return None

    monthly_amt = _group_monthly_amounts(txf)
    dataset_end = pd.to_datetime(transactions_df["date"]).max().to_period("M")
    range_end = max(monthly_amt.index.max(), dataset_end)
    if end_period is not None:
        range_end = max(range_end, end_period)
    full_index = pd.period_range(start=monthly_amt.index.min(), end=range_end, freq="M")
    return monthly_amt.reindex(full_index, fill_value=0).cumsum()


def map_account_net_worth_to_months(net_worth_series, months_to_plot):
    """Map cumulative net worth to plot months (0 before account activity, then ffill)."""
    nw_map = {
        str(p): v for p, v in zip(net_worth_series.index, net_worth_series.values)
    }
    values = []
    last_val = 0.0
    for month in months_to_plot:
        if month in nw_map:
            last_val = nw_map[month]
        values.append(last_val)
    return values


def _reindex_monthly_series(monthly):
    if monthly is None or len(monthly) == 0:
        return None
    full_month_index = pd.period_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="M",
    )
    return monthly.reindex(full_month_index, fill_value=0)


def build_entity_monthly_series(
    plot_type, name, transactions_df=None, abs_values=False
):
    """Monthly totals from full transaction history (ignores sidebar date range)."""
    if transactions_df is None:
        transactions_df = st.session_state.get("transactions_df")
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    col = "category_group" if plot_type == "category_group" else "category"
    ft = transactions_df[transactions_df[col] == name]
    if ft.empty:
        return None

    series = _reindex_monthly_series(_group_monthly_amounts(ft))
    if series is None:
        return None
    return abs(series) if abs_values else series


def filter_income_transactions(df):
    """Return inflow transactions excluding Starting Balance."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    return df[
        (df["category"] == "Inflow: Ready to Assign")
        & (df["payee_name"] != "Starting Balance")
    ].copy()


def filter_categorized_expense_transactions(df):
    """Return transactions that belong to a non-empty category group."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    return df[
        (df["category_group"].astype(str) != "nan")
        & (df["category_group"].astype(str) != "")
    ].copy()


def resolve_selected_category_groups(selected_category_groups=None):
    """Resolve active category groups from session state or fallback to all non-excluded."""
    if selected_category_groups is None:
        selected_category_groups = st.session_state.get("selected_category_groups", [])
    if selected_category_groups:
        return selected_category_groups
    excluded_groups = set(get_excluded_groups())
    category_groups = st.session_state.get("category_groups") or {}
    return sorted(g for g in category_groups if g not in excluded_groups)


def filter_transactions_by_selection(
    transactions_df,
    selected_category_groups,
    selected_categories,
    start_date=None,
    end_date=None,
    preserve_income=False,
):
    """Filter transactions by excluded groups, optional date range, and selected categories."""
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return pd.DataFrame()

    excluded_groups = get_excluded_groups()
    df = transactions_df[
        ~transactions_df["category_group"].isin(excluded_groups)
    ].copy()

    if start_date is not None and end_date is not None:
        df = filter_data_by_date_range(df, start_date, end_date)
    if df.empty:
        return df

    expense_df = df[
        (df["category_group"].isin(selected_category_groups))
        & (df["category"].isin(selected_categories))
    ]
    if not preserve_income:
        return expense_df.copy()

    income_df = filter_income_transactions(df)
    expense_only_df = expense_df[expense_df["category"] != "Inflow: Ready to Assign"]
    return pd.concat([income_df, expense_only_df], ignore_index=True)


def filter_budget_by_selection(
    budget_df, selected_category_groups, selected_categories
):
    """Filter budget DataFrame to selected category groups and categories."""
    if not isinstance(budget_df, pd.DataFrame) or budget_df.empty:
        return budget_df
    return budget_df[
        (budget_df["category_group"].isin(selected_category_groups))
        & (budget_df["category"].isin(selected_categories))
    ].copy()


def filter_overview_transactions_for_ma(transactions_df=None):
    """Full history with overview sidebar category filters (no date filter)."""
    if transactions_df is None:
        transactions_df = st.session_state.get("transactions_df")
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return transactions_df

    selected_groups = resolve_selected_category_groups()
    selected_categories = st.session_state.get("selected_categories", [])
    return filter_transactions_by_selection(
        transactions_df,
        selected_groups,
        selected_categories,
        preserve_income=True,
    )


def _monthly_series_or_empty(sub_df):
    """Group non-empty DataFrame by month or return an empty float Series."""
    return (
        _group_monthly_amounts(sub_df) if not sub_df.empty else pd.Series(dtype=float)
    )


def compute_overview_monthly_totals(df, data_type, abs_expense=False):
    """Compute monthly Series for total_income, total_expense, or total_net_income."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)

    if data_type == "total_income":
        return _monthly_series_or_empty(filter_income_transactions(df))

    if data_type == "total_expense":
        monthly = _monthly_series_or_empty(filter_categorized_expense_transactions(df))
        return monthly.abs() if abs_expense else monthly

    if data_type == "total_net_income":
        inc = _monthly_series_or_empty(filter_income_transactions(df))
        exp = _monthly_series_or_empty(filter_categorized_expense_transactions(df))
        return inc.add(exp, fill_value=0).sort_index()

    return pd.Series(dtype=float)


def build_overview_monthly_series(data_type, transactions_df=None):
    """Monthly totals from full history for overview moving averages."""
    if transactions_df is None:
        transactions_df = filter_overview_transactions_for_ma()
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    monthly = compute_overview_monthly_totals(
        transactions_df, data_type, abs_expense=(data_type == "total_expense")
    )
    if monthly.empty:
        return None
    return _reindex_monthly_series(monthly)


def _weighted_mean(values):
    """Linearly weighted mean where the most recent month has the highest weight."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    weights = np.arange(1, len(arr) + 1, dtype=float)
    return float(np.average(arr, weights=weights))


def calculate_series_tail_average(series, window):
    """Calculate absolute average of the tail of a monthly series for a given window."""
    if series is None or len(series) == 0:
        return 0
    if window == "weighted":
        return abs(_weighted_mean(series.tail(12)))
    return abs(series.tail(window).mean())


def moving_average_for_plot_months(full_monthly_series, months_to_plot, window=None):
    """Moving average from full history, values for selected plot months only."""
    if full_monthly_series is None or len(full_monthly_series) < 2:
        return None
    window = window or get_moving_average_window()
    ma_full = calculate_moving_average(full_monthly_series, window=window)
    return map_moving_average_to_months(ma_full, months_to_plot)


def calculate_moving_average(data, window=12):
    """Calculate moving average for the given data"""
    if len(data) < 2:
        return data

    if window == "weighted":
        actual_window = min(12, len(data))
        return data.rolling(window=actual_window, min_periods=1).apply(
            _weighted_mean, raw=True
        )

    actual_window = min(window, len(data))
    return data.rolling(window=actual_window, min_periods=1).mean()


def calculate_forecast_trend(data, periods=config.FORECAST_PERIODS):
    """Calculate forecast trend using linear regression"""
    if len(data) < 2:
        return data

    x = np.arange(len(data))
    coeffs = np.polyfit(x, data.values, 1)
    trend_line = np.polyval(coeffs, x)

    future_x = np.arange(len(data), len(data) + periods)
    forecast = np.polyval(coeffs, future_x)

    return pd.Series(trend_line, index=data.index), pd.Series(forecast, index=future_x)


def get_full_month_period_index(transactions_df):
    """Return a monthly PeriodIndex spanning the full date range of transactions_df."""
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None
    dates = pd.to_datetime(transactions_df["date"])
    return pd.period_range(
        start=dates.min().to_period("M"),
        end=dates.max().to_period("M"),
        freq="M",
    )


def calculate_series_window_averages(series, windows=AVERAGE_WINDOWS):
    """Calculate tail averages for each window in windows (default 3/6/12/24 months)."""
    return {w: calculate_series_tail_average(series, w) for w in windows}


def _resolve_unfiltered_transactions(fallback_df=None):
    """Return full unfiltered transactions DataFrame from session state or fallback."""
    full_df = st.session_state.get("transactions_df")
    if isinstance(full_df, pd.DataFrame) and not full_df.empty:
        return full_df
    return fallback_df


def _build_entity_monthly_expenses(
    column, name, transactions_df, global_month_range=None
):
    """Build monthly expense series for a category or group, zero-filled across full months."""
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    entity_tx = transactions_df[transactions_df[column] == name]
    if entity_tx.empty:
        return None

    monthly_expenses = _group_monthly_amounts(entity_tx)
    if monthly_expenses.empty:
        return None

    if global_month_range is not None and len(global_month_range) > 0:
        month_index = pd.PeriodIndex(global_month_range.strftime("%Y-%m"), freq="M")
    else:
        month_index = get_full_month_period_index(transactions_df)

    if month_index is not None and len(month_index) > 0:
        monthly_expenses = monthly_expenses.reindex(month_index, fill_value=0)
    return monthly_expenses


def _calculate_entity_averages(
    column, name, transactions_df, months=12, global_month_range=None
):
    """Shared average spending calculation for a category or category group."""
    monthly_expenses = _build_entity_monthly_expenses(
        column, name, transactions_df, global_month_range
    )
    return calculate_series_tail_average(monthly_expenses, months)


def calculate_entity_window_averages(
    plot_type, name, transactions_df=None, windows=AVERAGE_WINDOWS
):
    """Calculate 3/6/12/24-month averages from full unfiltered transaction history."""
    tx_df = _resolve_unfiltered_transactions(transactions_df)
    column = "category_group" if plot_type == "category_group" else "category"
    monthly_expenses = _build_entity_monthly_expenses(column, name, tx_df)
    return calculate_series_window_averages(monthly_expenses, windows)


def calculate_category_averages(
    category_name, transactions_df, months=12, global_month_range=None
):
    """Calculate average spending for a category over the last N months, including months with zero expenses"""
    return _calculate_entity_averages(
        "category", category_name, transactions_df, months, global_month_range
    )


def calculate_category_group_averages(group_name, transactions_df, months=3):
    """Calculate average spending for a category group over the last N months"""
    return _calculate_entity_averages(
        "category_group", group_name, transactions_df, months
    )


def calculate_category_available_budget(category_name, budget_df):
    """Calculate available budget for a category"""
    if not isinstance(budget_df, pd.DataFrame) or budget_df.empty:
        return 0
    cat_budget = budget_df[budget_df["category"] == category_name]
    return cat_budget["available"].iloc[0] if not cat_budget.empty else 0


def calculate_category_group_available_budget(group_name, budget_df):
    """Calculate total available budget for a category group"""
    if not isinstance(budget_df, pd.DataFrame) or budget_df.empty:
        return 0
    group_budget = budget_df[budget_df["category_group"] == group_name]
    return group_budget["available"].sum() if not group_budget.empty else 0


def month_start(d):
    """First calendar day of the month containing d."""
    ts = pd.Timestamp(d)
    return date(int(ts.year), int(ts.month), 1)


def _subtract_months(month_date, months):
    """Return the first-of-month date `months` calendar months before `month_date`."""
    total_months = (month_date.year * 12 + (month_date.month - 1)) - months
    year, month_zero = divmod(total_months, 12)
    return date(year, month_zero + 1, 1)


def month_end(d):
    """Last calendar day of the month containing d."""
    return (pd.Timestamp(month_start(d)) + pd.offsets.MonthEnd(0)).date()


def list_available_months(transactions_df=None):
    """First-of-month dates for every month in the data (or 2010 through today)."""
    today = date.today()
    if isinstance(transactions_df, pd.DataFrame) and not transactions_df.empty:
        dates = pd.to_datetime(transactions_df["date"])
        range_start = dates.min().replace(day=1)
        range_end = dates.max()
    else:
        range_start = pd.Timestamp(2010, 1, 1)
        range_end = pd.Timestamp(today)
    return [
        m.date() for m in pd.date_range(start=range_start, end=range_end, freq="MS")
    ]


def format_month_option(month_start_date):
    return month_start_date.strftime("%B %Y")


def dates_from_month_selection(start_month, end_month):
    """Inclusive bounds for a start month and end month (both first-of-month)."""
    return month_start(start_month), month_end(end_month)


def filter_data_by_date_range(transactions_df, start_date, end_date):
    """Filter transactions to full months (start month through end month inclusive)."""
    if transactions_df.empty:
        return transactions_df

    range_start = pd.Timestamp(month_start(start_date))
    range_end = pd.Timestamp(month_end(end_date))

    filtered_df = transactions_df.copy()
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])

    mask = (filtered_df["date"] >= range_start) & (filtered_df["date"] <= range_end)
    return filtered_df[mask]


def safe_strftime(dt):
    """Safely format datetime objects for display"""
    try:
        if pd.isna(dt):
            return "N/A"
        return dt.strftime("%Y-%m") if hasattr(dt, "strftime") else str(dt)
    except (ValueError, TypeError, AttributeError):
        return str(dt)


MONTH_RANGE_PRESET_OPTIONS = [
    "Last 3 Months",
    "Last 6 Months",
    "Last 12 Months",
    "Year to Date",
    "Last Year",
    "All Dates",
    "Custom",
]
DEFAULT_MONTH_RANGE_PRESET = "Last 12 Months"

_PRESET_MONTH_OFFSETS = {
    "Last 3 Months": 2,
    "Last 6 Months": 5,
    "Last 12 Months": 11,
}


def get_default_end_month():
    """Latest month included in presets (current month from the 15th, else prior month)."""
    today = date.today()
    if today.day >= 15:
        return month_start(today)
    first_day_current_month = date(today.year, today.month, 1)
    return month_start(first_day_current_month - timedelta(days=1))


def clamp_month_range(start_month, end_month, available_months):
    if not available_months:
        return start_month, end_month
    start_month = max(month_start(start_month), available_months[0])
    end_month = min(month_start(end_month), available_months[-1])
    return start_month, max(start_month, end_month)


def _resolve_bounded_end_month(available_months):
    """Determine preset end month clamped to available months."""
    end_month = get_default_end_month()
    if available_months:
        end_month = min(month_start(end_month), available_months[-1])
        end_month = max(end_month, available_months[0])
    return end_month


def _compute_preset_unclamped_bounds(preset, end_month):
    """Compute (start_month, end_month) before clamping to available_months."""
    if preset in _PRESET_MONTH_OFFSETS:
        offset = _PRESET_MONTH_OFFSETS[preset]
        return _subtract_months(end_month, offset), end_month
    if preset == "Year to Date":
        return date(end_month.year, 1, 1), end_month
    if preset == "Last Year":
        year = end_month.year - 1
        return date(year, 1, 1), date(year, 12, 1)
    raise ValueError(f"Unknown preset: {preset}")


def resolve_month_range_preset(preset, available_months):
    """Return (start_month, end_month) as first-of-month dates for a sidebar preset."""
    if preset == "All Dates":
        if available_months:
            return available_months[0], available_months[-1]
        end_month = get_default_end_month()
        return _subtract_months(end_month, 11), end_month

    end_month = _resolve_bounded_end_month(available_months)
    start_month, end_month = _compute_preset_unclamped_bounds(preset, end_month)
    return clamp_month_range(start_month, end_month, available_months)


def get_default_month_range():
    """Default display range: last 12 calendar months (first-of-month for each bound)."""
    return resolve_month_range_preset("Last 12 Months", [])


def get_default_date_range():
    """Backward-compatible inclusive date bounds from default month range."""
    start_month, end_month = get_default_month_range()
    return dates_from_month_selection(start_month, end_month)


def get_global_month_range(transactions_df, start_date, end_date):
    """Month range for plots, aligned to the sidebar month selection."""
    earliest_date = pd.Timestamp(month_start(start_date))
    latest_date = pd.Timestamp(month_start(end_date))
    global_month_range = pd.date_range(start=earliest_date, end=latest_date, freq="MS")
    return global_month_range, earliest_date, latest_date


def get_excluded_groups():
    """Get list of excluded category groups"""
    return [
        "Internal Master Category",
        "Uncategorized",
        "Credit Card Payments",
        "Hidden Categories",
    ]


def get_default_category_groups():
    """Get default category groups for filtering"""
    return ["Lazer", "Necessidades"]


def get_default_categories():
    """Get default categories for filtering"""
    return ["Groceries", "Dining Out", "Transportation"]


def _collect_entity_y_values(
    transactions_df, categories_data, column_name, target_key, names
):
    """Collect monthly expense and target y-values for categories or category groups"""
    y_values = []
    for name in names:
        filtered_tx = transactions_df[transactions_df[column_name] == name]
        if filtered_tx.empty:
            continue
        monthly_expenses = _group_monthly_amounts(filtered_tx)
        y_values.extend(abs(monthly_expenses.values))
        y_values.extend(
            cat["target_amount"]
            for cat in categories_data
            if cat[target_key] == name and cat.get("target_amount") is not None
        )
    return y_values


def _collect_overview_y_values(transactions_df):
    """Collect monthly income and expense y-values for overview plots"""
    inc = compute_overview_monthly_totals(transactions_df, "total_income")
    exp = compute_overview_monthly_totals(
        transactions_df, "total_expense", abs_expense=True
    )
    return list(inc.values) + list(exp.values)


def calculate_global_y_range(
    transactions_df,
    budget_df,
    categories_data,
    plot_type="category",
    category_names=None,
):
    """Calculate global y-axis range for a set of plots."""
    if transactions_df.empty:
        return None

    if plot_type == "overview":
        all_y_values = _collect_overview_y_values(transactions_df)
    elif category_names and plot_type in ("category", "category_group"):
        target_key = "name" if plot_type == "category" else "group"
        all_y_values = _collect_entity_y_values(
            transactions_df, categories_data, plot_type, target_key, category_names
        )
    else:
        all_y_values = []

    if not all_y_values:
        return None

    min_val, max_val = min(all_y_values), max(all_y_values)
    padding = (max_val - min_val) * 0.1
    return [max(0, min_val - padding), max_val + padding]


def format_target_delta(value, target_amount, zero_text=None):
    """Calculate percentage delta text and color against a target budget."""
    if not target_amount or target_amount <= 0:
        return None, "normal"
    delta = value - target_amount
    if delta == 0 and zero_text is not None:
        return zero_text, "inverse"
    delta_pct = (delta / target_amount) * 100
    return f"{delta_pct:+.1f}%", "inverse"


def build_complete_monthly_df(global_month_range, monthly_df=None):
    """Build a complete month DataFrame aligned to global_month_range."""
    all_months_df = pd.DataFrame(
        {
            "month_date": global_month_range,
            "month": global_month_range.strftime("%Y-%m"),
        }
    )
    if monthly_df is None or monthly_df.empty:
        all_months_df["amount"] = 0.0
        return all_months_df.sort_values("month_date")

    return (
        all_months_df.merge(monthly_df[["month", "amount"]], on="month", how="left")
        .fillna(0)
        .sort_values("month_date")
    )


def add_bar_trace(fig, x_months, y_values, name, color, hover_label=None):
    """Add a formatted currency Bar trace to a Plotly figure."""
    label = hover_label or name
    fig.add_trace(
        go.Bar(
            x=x_months,
            y=y_values,
            name=name,
            marker_color=color,
            opacity=0.8,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{customdata}}<br><extra></extra>",
            customdata=[format_currency(val) for val in y_values],
        )
    )


def add_moving_average_trace(fig, months_to_plot, ma_plot, ma_window=None):
    """Add a formatted moving average Scatter trace to a Plotly figure."""
    if ma_plot is None:
        return
    ma_label = moving_average_label(ma_window or get_moving_average_window())
    fig.add_trace(
        go.Scatter(
            x=months_to_plot,
            y=ma_plot,
            name=ma_label,
            line={"color": "#1f77b4", "width": 2, "dash": "dot"},
            mode="lines",
            hovertemplate=f"<b>%{{x}}</b><br>{ma_label}: %{{customdata}}<br><extra></extra>",
            customdata=[format_currency(v) if v is not None else "€0" for v in ma_plot],
        )
    )


def add_forecast_traces(
    fig,
    months_series,
    month_dates_series,
    values_for_regression,
    abs_forecast=True,
    trend_name="12-Month Forecast Trend",
):
    """Add linear regression trend line and 3-month forecast extension traces."""
    if len(values_for_regression) < 3:
        return None

    trend_line, forecast = calculate_forecast_trend(values_for_regression)
    fig.add_trace(
        go.Scatter(
            x=months_series,
            y=trend_line,
            name=trend_name,
            line={"color": "#d62728", "width": 2},
            mode="lines",
            hovertemplate=f"<b>%{{x}}</b><br>{trend_name}: %{{customdata}}<br><extra></extra>",
            customdata=[format_currency(val) for val in trend_line],
        )
    )

    last_month_date = month_dates_series.iloc[-1]
    future_months = pd.date_range(
        start=last_month_date + pd.DateOffset(months=1),
        periods=config.FORECAST_PERIODS,
        freq="MS",
    )
    forecast_values = abs(forecast) if abs_forecast else forecast
    fig.add_trace(
        go.Scatter(
            x=future_months.strftime("%Y-%m"),
            y=forecast_values,
            name="Forecast (Next 3 Months)",
            line={"color": "#d62728", "width": 2, "dash": "dash"},
            mode="lines",
            hovertemplate="<b>%{x}</b><br>Forecast: %{customdata}<br><extra></extra>",
            customdata=[format_currency(val) for val in forecast_values],
        )
    )
    return future_months


def add_target_goal_trace(fig, base_months, future_months, target_amount):
    """Add a horizontal target goal line across historical and forecast months."""
    if target_amount is None or target_amount <= 0:
        return
    all_months = list(base_months)
    if future_months is not None:
        all_months.extend(future_months.strftime("%Y-%m").tolist())
    formatted_target = format_currency(target_amount)
    fig.add_trace(
        go.Scatter(
            x=all_months,
            y=[target_amount] * len(all_months),
            name=f"Target Goal ({formatted_target})",
            line={"color": "#2ca02c", "width": 3, "dash": "solid"},
            mode="lines",
            hovertemplate="<b>%{x}</b><br>Target Goal: %{customdata}<br><extra></extra>",
            customdata=[formatted_target] * len(all_months),
        )
    )


def apply_standard_plot_layout(fig, height=400, y_range=None, yaxis_title="Amount (€)"):
    """Apply consistent layout settings to a Plotly figure."""
    layout_update = {
        "height": height,
        "showlegend": False,
        "hovermode": "x unified",
        "xaxis_title": "Month",
        "yaxis_title": yaxis_title,
        "barmode": "overlay",
    }
    if y_range is not None:
        layout_update["yaxis"] = {"range": y_range}
    fig.update_layout(**layout_update)
    return fig


def render_average_metrics_grid(averages, target_amount=None, short=True):
    """Render a 2x2 grid of 3, 6, 12, and 24-month average metrics."""
    unit = "Avg" if short else "Average"
    for row_windows in ((3, 6), (12, 24)):
        cols = st.columns(2)
        for col, window in zip(cols, row_windows):
            with col:
                val = averages.get(window, 0)
                delta_text, delta_color = format_target_delta(
                    val, target_amount, zero_text="On target"
                )
                st.metric(
                    label=f"📊 Last {window} Months {unit}",
                    value=format_currency(val),
                    delta=delta_text,
                    delta_color=delta_color,
                    help=f"Average monthly amount over the last {window} months",
                )


def _prepare_unified_plot_data(
    name, transactions_df, budget_df, categories_data, plot_type
):
    """Filter data and calculate summary metrics for unified plot"""
    if plot_type == "category_group":
        filtered_transactions = transactions_df[
            transactions_df["category_group"] == name
        ].copy()
        target_amount = sum(
            cat["target_amount"]
            for cat in categories_data
            if cat["group"] == name and cat.get("target_amount") is not None
        )
        available_budget = calculate_category_group_available_budget(name, budget_df)
    else:
        filtered_transactions = transactions_df[
            transactions_df["category"] == name
        ].copy()
        target_amount = next(
            (
                cat["target_amount"]
                for cat in categories_data
                if cat["name"] == name and cat.get("target_amount") is not None
            ),
            None,
        )
        available_budget = calculate_category_available_budget(name, budget_df)

    if filtered_transactions.empty:
        return None

    averages = calculate_entity_window_averages(plot_type, name, transactions_df)
    return filtered_transactions, target_amount, averages, available_budget


def _render_unified_plot_metrics(target_amount, averages, available_budget):
    """Render Streamlit metrics for a unified category or category group plot"""
    col1, col2 = st.columns(2)
    with col1:
        target_val = (
            format_currency(target_amount)
            if (target_amount is not None and target_amount > 0)
            else "No target set"
        )
        st.metric(label="🎯 Target Budget", value=target_val)
    with col2:
        st.metric(
            label="💰 Available Budget",
            value=format_currency(available_budget),
        )

    render_average_metrics_grid(averages, target_amount=target_amount, short=True)


def _add_unified_plot_traces(
    fig, complete_monthly_data, plot_type, name, target_amount
):
    """Add bar, moving average, forecast, and target goal traces to unified plot"""
    months_to_plot = complete_monthly_data["month"].tolist()
    abs_amounts = abs(complete_monthly_data["amount"])

    add_bar_trace(fig, months_to_plot, abs_amounts, "Actual Expenses", "#ff7f0e")

    ma_window = get_moving_average_window()
    full_series_abs = build_entity_monthly_series(plot_type, name, abs_values=True)
    ma_plot = moving_average_for_plot_months(
        full_series_abs, months_to_plot, window=ma_window
    )
    add_moving_average_trace(fig, months_to_plot, ma_plot, ma_window=ma_window)

    future_months = add_forecast_traces(
        fig,
        complete_monthly_data["month"],
        complete_monthly_data["month_date"],
        abs_amounts,
        abs_forecast=True,
    )
    add_target_goal_trace(fig, months_to_plot, future_months, target_amount)


def create_unified_plot(
    name,
    transactions_df,
    budget_df,
    global_month_range,
    categories_data,
    plot_type="category",
    y_range=None,
):
    """Create comprehensive plot for either a category group or individual category."""
    prepared = _prepare_unified_plot_data(
        name, transactions_df, budget_df, categories_data, plot_type
    )
    if prepared is None:
        return None

    filtered_transactions, target_amount, averages, available_budget = prepared

    _render_unified_plot_metrics(target_amount, averages, available_budget)

    monthly_series = _group_monthly_amounts(filtered_transactions)
    monthly_expenses = monthly_series.reset_index()
    monthly_expenses["month"] = monthly_expenses["month"].astype(str)

    fig = go.Figure()
    if not monthly_expenses.empty:
        complete_monthly_data = build_complete_monthly_df(
            global_month_range, monthly_expenses
        )
        _add_unified_plot_traces(
            fig, complete_monthly_data, plot_type, name, target_amount
        )

    return apply_standard_plot_layout(fig, height=400, y_range=y_range)
