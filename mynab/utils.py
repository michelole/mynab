import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from ynab.configuration import Configuration
from ynab.api_client import ApiClient
from ynab.api import PlansApi, CategoriesApi, TransactionsApi, MonthsApi, AccountsApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def format_currency(amount):
    """Format currency amount with no cents and period as thousand separator"""
    if pd.isna(amount) or amount is None:
        return "€0"

    # Round to nearest integer (no cents)
    rounded_amount = round(amount)

    # Format with period as thousand separator
    if rounded_amount >= 0:
        return f"€{rounded_amount:,}".replace(",", ".")
    else:
        return f"-€{abs(rounded_amount):,}".replace(",", ".")


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ynab_data(access_token):
    """Fetch data from YNAB API"""
    try:
        configuration = Configuration(access_token=access_token)
        with ApiClient(configuration) as api_client:
            plans_api = PlansApi(api_client)
            plans_response = plans_api.get_plans()

            if not plans_response.data.plans:
                st.error("No plans found in your YNAB account.")
                return None, None, None, None, None, None

            plan = plans_response.data.plans[0]
            budget_id = str(plan.id)
            budget_name = plan.name

            categories_api = CategoriesApi(api_client)
            categories_response = categories_api.get_categories(budget_id)

            transactions_api = TransactionsApi(api_client)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)

            transactions_response = transactions_api.get_transactions(
                budget_id, since_date=start_date.date()
            )

            months_api = MonthsApi(api_client)
            current_month = datetime.now().replace(day=1).date()
            months_response = months_api.get_plan_month(budget_id, current_month)

            accounts_api = AccountsApi(api_client)
            accounts_response = accounts_api.get_accounts(budget_id)
            accounts_list = []
            if accounts_response.data.accounts:
                for acc in accounts_response.data.accounts:
                    acc_type = getattr(acc, "type", None)
                    accounts_list.append(
                        {
                            "id": str(acc.id),
                            "name": acc.name,
                            "type": acc_type.value if acc_type is not None else None,
                            "on_budget": getattr(acc, "on_budget", None),
                            "closed": getattr(acc, "closed", None),
                        }
                    )

            return (
                budget_id,
                budget_name,
                categories_response,
                transactions_response,
                months_response,
                accounts_list,
            )

    except Exception as e:
        st.error(f"Error connecting to YNAB API: {str(e)}")
        return None, None, None, None, None, None


def process_categories_data(categories_response):
    """Process categories data into a structured format, excluding specified groups"""
    categories_data = []
    category_groups = {}

    # Categories to exclude
    excluded_groups = [
        "Internal Master Category",
        "Uncategorized",
        "Credit Card Payments",
        "Hidden Categories",
    ]

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

                if hasattr(category, "goal_target") and category.goal_target:
                    target_amount = (
                        category.goal_target / 1000
                    )  # Convert from millidollars

                if hasattr(category, "goal_type") and category.goal_type:
                    target_type = category.goal_type

                if (
                    hasattr(category, "goal_target_month")
                    and category.goal_target_month
                ):
                    target_date = category.goal_target_month

                category_data = {
                    "id": category.id,
                    "name": category.name,
                    "group": group_name,
                    "category_group_id": group.id,
                    "target_amount": target_amount,
                    "target_type": target_type,
                    "target_date": target_date,
                }
                categories_data.append(category_data)
                category_groups[group_name].append(category_data)

    return categories_data, category_groups


def _find_category_group(category_id, categories_data):
    """Find category group name for a given category ID"""
    if not category_id:
        return ""
    for cat in categories_data:
        if cat["id"] == category_id:
            return cat["group"]
    return ""


def _get_transaction_date(transaction):
    """Extract transaction date from YNAB transaction object"""
    if hasattr(transaction, "var_date"):
        return transaction.var_date
    if hasattr(transaction, "date"):
        return transaction.date
    return None


def _build_subtransaction_record(
    subtransaction, transaction, transaction_date, account_id, account_name, categories_data
):
    """Build a dictionary record for a split subtransaction"""
    category_group = _find_category_group(subtransaction.category_id, categories_data)
    return {
        "date": transaction_date,
        "amount": subtransaction.amount / 1000,
        "category": subtransaction.category_name or "",
        "category_group": category_group,
        "payee_name": transaction.payee_name or "",
        "memo": subtransaction.memo or transaction.memo or "",
        "transaction_id": transaction.id,
        "account_id": account_id,
        "account_name": account_name,
        "import_payee_name": getattr(
            subtransaction,
            "import_payee_name",
            getattr(transaction, "import_payee_name", ""),
        ),
        "import_payee_name_original": getattr(
            subtransaction,
            "import_payee_name_original",
            getattr(transaction, "import_payee_name_original", ""),
        ),
    }


def _build_transaction_record(
    transaction, transaction_date, account_id, account_name, categories_data
):
    """Build a dictionary record for a regular non-split transaction"""
    category_group = _find_category_group(transaction.category_id, categories_data)
    return {
        "date": transaction_date,
        "amount": transaction.amount / 1000,
        "category": transaction.category_name or "",
        "category_group": category_group,
        "payee_name": transaction.payee_name or "",
        "memo": transaction.memo or "",
        "transaction_id": transaction.id,
        "account_id": account_id,
        "account_name": account_name,
        "import_payee_name": getattr(transaction, "import_payee_name", ""),
        "import_payee_name_original": getattr(
            transaction, "import_payee_name_original", ""
        ),
    }


def process_transactions_data(transactions_response, categories_data):
    """Process transactions data into a structured format, handling split transactions"""
    transactions = []

    for transaction in transactions_response.data.transactions:
        if transaction.amount == 0:
            continue

        transaction_date = _get_transaction_date(transaction)
        if transaction_date is None:
            continue

        account_id = getattr(transaction, "account_id", None)
        account_name = getattr(transaction, "account_name", None)
        subtransactions = getattr(transaction, "subtransactions", None)

        if subtransactions:
            for subtransaction in subtransactions:
                if subtransaction.amount != 0:
                    transactions.append(
                        _build_subtransaction_record(
                            subtransaction,
                            transaction,
                            transaction_date,
                            account_id,
                            account_name,
                            categories_data,
                        )
                    )
        else:
            transactions.append(
                _build_transaction_record(
                    transaction,
                    transaction_date,
                    account_id,
                    account_name,
                    categories_data,
                )
            )

    df = pd.DataFrame(transactions)

    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def process_months_data(months_response, categories_data):
    """Process months data for budget information"""
    budget_data = []

    if months_response and months_response.data.month:
        month_data = months_response.data.month

        # Extract category budgets from the current month
        if hasattr(month_data, "categories"):
            for category in month_data.categories:
                if category.budgeted != 0 or category.activity != 0:
                    # Find category name and target information
                    category_name = "Uncategorized"
                    category_group = "Uncategorized"
                    target_amount = None
                    target_type = None
                    target_date = None

                    for cat in categories_data:
                        if cat["id"] == category.id:
                            category_name = cat["name"]
                            category_group = cat["group"]
                            target_amount = cat.get("target_amount")
                            target_type = cat.get("target_type")
                            target_date = cat.get("target_date")
                            break

                    budget_data.append(
                        {
                            "category": category_name,
                            "category_group": category_group,
                            "budgeted": category.budgeted
                            / 1000,  # Convert from millidollars
                            "activity": category.activity
                            / 1000,  # Convert from millidollars
                            "available": category.balance
                            / 1000,  # Convert from millidollars
                            "target_amount": target_amount,
                            "target_type": target_type,
                            "target_date": target_date,
                        }
                    )

    df = pd.DataFrame(budget_data)
    return df


MOVING_AVERAGE_WINDOW_OPTIONS = {
    "12 months (weighted)": "weighted",
    "3 months": 3,
    "6 months": 6,
    "12 months": 12,
    "24 months": 24,
}
DEFAULT_MOVING_AVERAGE_WINDOW = "weighted"


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


def build_account_net_worth_series(
    account_name, transactions_df=None, end_period=None
):
    """Cumulative net worth by month from full account history."""
    if transactions_df is None:
        transactions_df = st.session_state.get("transactions_df")
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    txf = transactions_df[transactions_df["account_name"] == account_name].copy()
    if txf.empty:
        return None

    txf["date"] = pd.to_datetime(txf["date"])
    txf["month"] = txf["date"].dt.to_period("M")
    monthly_amt = txf.groupby("month")["amount"].sum().sort_index()
    range_end = monthly_amt.index.max()
    if end_period is not None:
        range_end = max(range_end, end_period)
    full_index = pd.period_range(
        start=monthly_amt.index.min(), end=range_end, freq="M"
    )
    monthly_amt = monthly_amt.reindex(full_index, fill_value=0)
    return monthly_amt.cumsum()


def map_account_net_worth_to_months(net_worth_series, months_to_plot):
    """Map cumulative net worth to plot months (0 before account activity, then ffill)."""
    nw_map = {str(p): v for p, v in zip(net_worth_series.index, net_worth_series.values)}
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

    ft = transactions_df.copy()
    ft["date"] = pd.to_datetime(ft["date"])
    ft["month"] = ft["date"].dt.to_period("M")
    if plot_type == "category_group":
        ft = ft[ft["category_group"] == name]
    else:
        ft = ft[ft["category"] == name]
    monthly = ft.groupby("month")["amount"].sum().sort_index()
    series = _reindex_monthly_series(monthly)
    if series is None:
        return None
    return abs(series) if abs_values else series


def filter_overview_transactions_for_ma(transactions_df=None):
    """Full history with overview sidebar category filters (no date filter)."""
    if transactions_df is None:
        transactions_df = st.session_state.get("transactions_df")
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return transactions_df

    excluded_groups = get_excluded_groups()
    df = transactions_df[
        ~transactions_df["category_group"].isin(excluded_groups)
    ].copy()

    selected_category_groups = st.session_state.get("selected_category_groups", [])
    if not selected_category_groups:
        category_groups = st.session_state.get("category_groups", {})
        selected_category_groups = sorted(
            group
            for group in category_groups.keys()
            if group not in excluded_groups
        )

    selected_categories = st.session_state.get("selected_categories", [])

    income_transactions = df[
        (df["category"] == "Inflow: Ready to Assign")
        & (df["payee_name"] != "Starting Balance")
    ].copy()
    expense_transactions = df[
        (df["category_group"].isin(selected_category_groups))
        & (df["category"].isin(selected_categories))
        & (df["category"] != "Inflow: Ready to Assign")
    ].copy()

    return pd.concat([income_transactions, expense_transactions], ignore_index=True)


def build_overview_monthly_series(data_type, transactions_df=None):
    """Monthly totals from full history for overview moving averages."""
    if transactions_df is None:
        transactions_df = filter_overview_transactions_for_ma()
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    if data_type == "total_income":
        df = df[
            (df["category"] == "Inflow: Ready to Assign")
            & (df["payee_name"] != "Starting Balance")
        ]
        monthly = df.groupby("month")["amount"].sum()
    elif data_type == "total_expense":
        df = df[
            (df["category_group"].astype(str) != "nan")
            & (df["category_group"].astype(str) != "")
        ]
        monthly = df.groupby("month")["amount"].sum().abs()
    elif data_type == "total_net_income":
        income_df = df[
            (df["category"] == "Inflow: Ready to Assign")
            & (df["payee_name"] != "Starting Balance")
        ]
        expense_df = df[
            (df["category_group"].astype(str) != "nan")
            & (df["category_group"].astype(str) != "")
        ]
        monthly_income = income_df.groupby("month")["amount"].sum()
        monthly_expenses = expense_df.groupby("month")["amount"].sum()
        all_months = pd.concat([monthly_income, monthly_expenses]).index.unique()
        monthly = pd.Series(index=all_months, dtype=float)
        for month in all_months:
            monthly[month] = monthly_income.get(month, 0) + monthly_expenses.get(
                month, 0
            )
    else:
        return None

    return _reindex_monthly_series(monthly.sort_index())


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


def calculate_category_averages(
    category_name, transactions_df, months=12, global_month_range=None
):
    """Calculate average spending for a category over the last N months, including months with zero expenses"""
    # Filter data for this category
    cat_transactions = transactions_df[
        transactions_df["category"] == category_name
    ].copy()

    if cat_transactions.empty:
        # If no transactions, but global_month_range is provided, return 0
        if global_month_range is not None and len(global_month_range) > 0:
            return 0
        else:
            return 0

    # Convert date and group by month
    cat_transactions["date"] = pd.to_datetime(cat_transactions["date"])
    cat_transactions["month"] = cat_transactions["date"].dt.to_period("M")

    # Get monthly totals
    monthly_expenses = cat_transactions.groupby("month")["amount"].sum()
    monthly_expenses = monthly_expenses.sort_index()

    is_weighted = months == "weighted"
    n_months = 12 if is_weighted else months

    # If global_month_range is provided, reindex to include all months
    if global_month_range is not None and len(global_month_range) > 0:
        # Convert global_month_range to PeriodIndex
        global_months = pd.PeriodIndex(global_month_range.strftime("%Y-%m"), freq="M")
        monthly_expenses = monthly_expenses.reindex(global_months, fill_value=0)
        # Use the last N months from the global range
        last_n_months = monthly_expenses[-n_months:]
    else:
        # Use the last N months from available data
        last_n_months = monthly_expenses.tail(n_months)

    # Calculate average (convert to positive for display)
    if is_weighted:
        avg_amount = abs(_weighted_mean(last_n_months))
    else:
        avg_amount = abs(last_n_months.mean())

    return avg_amount


def calculate_category_available_budget(category_name, budget_df):
    """Calculate available budget for a category"""
    if budget_df.empty:
        return 0

    # Filter budget data for this category
    cat_budget = budget_df[budget_df["category"] == category_name]

    if cat_budget.empty:
        return 0

    # Get the available amount for this category
    available = cat_budget["available"].iloc[0]

    return available


def calculate_category_group_averages(group_name, transactions_df, months=3):
    """Calculate average spending for a category group over the last N months"""
    # Filter data for this category group
    group_transactions = transactions_df[
        transactions_df["category_group"] == group_name
    ].copy()

    if group_transactions.empty:
        return 0

    # Convert date and group by month
    group_transactions["date"] = pd.to_datetime(group_transactions["date"])
    group_transactions["month"] = group_transactions["date"].dt.to_period("M")

    # Get monthly totals
    monthly_expenses = group_transactions.groupby("month")["amount"].sum()

    if monthly_expenses.empty:
        return 0

    is_weighted = months == "weighted"
    n_months = 12 if is_weighted else months

    # Sort by month and get the last N months
    monthly_expenses = monthly_expenses.sort_index()
    last_n_months = monthly_expenses.tail(n_months)

    # Calculate average (convert to positive for display)
    if is_weighted:
        avg_amount = abs(_weighted_mean(last_n_months))
    else:
        avg_amount = abs(last_n_months.mean())

    return avg_amount


def calculate_category_group_available_budget(group_name, budget_df):
    """Calculate total available budget for a category group"""
    if budget_df.empty:
        return 0

    # Filter budget data for this category group
    group_budget = budget_df[budget_df["category_group"] == group_name]

    if group_budget.empty:
        return 0

    # Sum the available amounts for all categories in this group
    total_available = group_budget["available"].sum()

    return total_available


def month_start(d):
    """First calendar day of the month containing d."""
    d = pd.Timestamp(d)
    return date(d.year, d.month, 1)


def month_end(d):
    """Last calendar day of the month containing d."""
    return (pd.Timestamp(month_start(d)) + pd.offsets.MonthEnd(0)).date()


def list_available_months(transactions_df=None):
    """First-of-month dates for every month in the data (or 2010 through today)."""
    today = date.today()
    if isinstance(transactions_df, pd.DataFrame) and not transactions_df.empty:
        df = transactions_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        range_start = df["date"].min().replace(day=1)
        range_end = df["date"].max()
    else:
        range_start = pd.Timestamp(2010, 1, 1)
        range_end = pd.Timestamp(today)
    return [m.date() for m in pd.date_range(start=range_start, end=range_end, freq="MS")]


def format_month_option(month_start_date):
    return month_start_date.strftime("%B %Y")


def dates_from_month_selection(start_month, end_month):
    """Inclusive bounds for a start month and end month (both first-of-month)."""
    return month_start(start_month), month_end(end_month)


def filter_data_by_date_range(transactions_df, start_date, end_date):
    """Filter transactions to full months (start month through end month inclusive)."""
    if transactions_df.empty:
        return transactions_df

    range_start = month_start(start_date)
    range_end = month_end(end_date)

    filtered_df = transactions_df.copy()
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])

    mask = (filtered_df["date"] >= pd.Timestamp(range_start)) & (
        filtered_df["date"] <= pd.Timestamp(range_end)
    )
    return filtered_df[mask]


def safe_strftime(dt):
    """Safely format datetime objects for display"""
    try:
        if pd.isna(dt):
            return "N/A"
        return dt.strftime("%Y-%m") if hasattr(dt, "strftime") else str(dt)
    except Exception:
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
    if start_month > end_month:
        end_month = start_month
    return start_month, end_month


def resolve_month_range_preset(preset, available_months):
    """Return (start_month, end_month) as first-of-month dates for a sidebar preset."""
    if preset == "All Dates":
        if not available_months:
            end_month = get_default_end_month()
            start_month = (pd.Timestamp(end_month) - pd.DateOffset(months=11)).date()
            return start_month, end_month
        return available_months[0], available_months[-1]

    end_month = get_default_end_month()
    if available_months:
        end_month = min(month_start(end_month), available_months[-1])
        end_month = max(end_month, available_months[0])

    if preset == "Last 3 Months":
        start_month = (pd.Timestamp(end_month) - pd.DateOffset(months=2)).date()
    elif preset == "Last 6 Months":
        start_month = (pd.Timestamp(end_month) - pd.DateOffset(months=5)).date()
    elif preset == "Last 12 Months":
        start_month = (pd.Timestamp(end_month) - pd.DateOffset(months=11)).date()
    elif preset == "Year to Date":
        start_month = date(end_month.year, 1, 1)
    elif preset == "Last Year":
        year = end_month.year - 1
        start_month = date(year, 1, 1)
        end_month = date(year, 12, 1)
    else:
        raise ValueError(f"Unknown preset: {preset}")

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
    global_month_range = pd.date_range(
        start=earliest_date, end=latest_date, freq="MS"
    )
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
        filtered_transactions = transactions_df[
            transactions_df[column_name] == name
        ].copy()

        if not filtered_transactions.empty:
            filtered_transactions["date"] = pd.to_datetime(
                filtered_transactions["date"]
            )
            filtered_transactions["month"] = filtered_transactions[
                "date"
            ].dt.to_period("M")
            monthly_expenses = filtered_transactions.groupby("month")["amount"].sum()
            y_values.extend(abs(monthly_expenses.values))

            for cat in categories_data:
                if (
                    cat[target_key] == name
                    and cat.get("target_amount") is not None
                ):
                    y_values.append(cat["target_amount"])
    return y_values


def _collect_overview_y_values(transactions_df):
    """Collect monthly income and expense y-values for overview plots"""
    y_values = []
    income_transactions = transactions_df[
        (transactions_df["category"] == "Inflow: Ready to Assign")
        & (transactions_df["payee_name"] != "Starting Balance")
    ].copy()

    if not income_transactions.empty:
        income_transactions["date"] = pd.to_datetime(income_transactions["date"])
        income_transactions["month"] = income_transactions["date"].dt.to_period("M")
        monthly_income = income_transactions.groupby("month")["amount"].sum()
        y_values.extend(monthly_income.values)

    expense_transactions = transactions_df[
        (transactions_df["category_group"].astype(str) != "nan")
        & (transactions_df["category_group"].astype(str) != "")
    ].copy()

    if not expense_transactions.empty:
        expense_transactions["date"] = pd.to_datetime(expense_transactions["date"])
        expense_transactions["month"] = expense_transactions["date"].dt.to_period(
            "M"
        )
        monthly_expenses = expense_transactions.groupby("month")["amount"].sum()
        y_values.extend(abs(monthly_expenses.values))

    return y_values


def calculate_global_y_range(
    transactions_df,
    budget_df,
    categories_data,
    plot_type="category",
    category_names=None,
):
    """
    Calculate global y-axis range for a set of plots

    Args:
        transactions_df: DataFrame of transactions
        budget_df: DataFrame of budget data
        categories_data: List of category data dictionaries
        plot_type: 'category', 'category_group', or 'overview'
        category_names: List of category/group names to include in calculation

    Returns:
        tuple: (min_value, max_value) for y-axis range
    """
    if transactions_df.empty:
        return None

    all_y_values = []

    if plot_type == "category" and category_names:
        all_y_values = _collect_entity_y_values(
            transactions_df, categories_data, "category", "name", category_names
        )
    elif plot_type == "category_group" and category_names:
        all_y_values = _collect_entity_y_values(
            transactions_df, categories_data, "category_group", "group", category_names
        )
    elif plot_type == "overview":
        all_y_values = _collect_overview_y_values(transactions_df)

    if all_y_values:
        min_val = min(all_y_values)
        max_val = max(all_y_values)
        padding = (max_val - min_val) * 0.1
        return [max(0, min_val - padding), max_val + padding]

    return None


def _prepare_unified_plot_data(
    name, transactions_df, budget_df, global_month_range, categories_data, plot_type
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
        avg_12_months = calculate_category_group_averages(name, transactions_df, 12)
        available_budget = calculate_category_group_available_budget(name, budget_df)
    else:
        filtered_transactions = transactions_df[
            transactions_df["category"] == name
        ].copy()
        cat_budget = pd.DataFrame()
        if not budget_df.empty and "category" in budget_df.columns:
            cat_budget = budget_df[budget_df["category"] == name].copy()
        if filtered_transactions.empty and cat_budget.empty:
            return None

        target_amount = next(
            (
                cat["target_amount"]
                for cat in categories_data
                if cat["name"] == name and cat.get("target_amount") is not None
            ),
            None,
        )
        avg_12_months = calculate_category_averages(
            name, transactions_df, 12, global_month_range
        )
        available_budget = calculate_category_available_budget(name, budget_df)

    if filtered_transactions.empty:
        return None

    return filtered_transactions, target_amount, avg_12_months, available_budget


def _render_unified_plot_metrics(
    name,
    transactions_df,
    global_month_range,
    plot_type,
    target_amount,
    avg_12_months,
    available_budget,
):
    """Render Streamlit metrics for a unified category or category group plot"""
    import streamlit as st

    col1, col2 = st.columns(2)

    with col1:
        if target_amount is not None and target_amount > 0:
            st.metric(
                label="🎯 Target Budget",
                value=format_currency(target_amount),
            )
        else:
            st.metric(label="🎯 Target Budget", value="No target set")

    with col2:
        st.metric(
            label="💰 Available Budget",
            value=format_currency(available_budget),
        )

    col3, col4 = st.columns(2)

    with col3:
        if target_amount is not None and target_amount > 0:
            delta_12m = avg_12_months - target_amount
            delta_pct_12m = (delta_12m / target_amount) * 100
            delta_color_12m = "inverse"
            delta_text_12m = f"{delta_pct_12m:+.1f}%" if delta_12m != 0 else "On target"
        else:
            delta_text_12m = None
            delta_color_12m = "normal"

        st.metric(
            label="📊 Last 12 Months Avg",
            value=format_currency(avg_12_months),
            delta=delta_text_12m,
            delta_color=delta_color_12m,
        )

    ma_window = get_moving_average_window()
    if plot_type == "category_group":
        last_ma_avg = calculate_category_group_averages(
            name, transactions_df, ma_window
        )
    else:
        last_ma_avg = calculate_category_averages(
            name, transactions_df, ma_window, global_month_range
        )

    with col4:
        suggested_budget = -(available_budget - (avg_12_months * 12)) / 12
        if target_amount is not None and target_amount > 0:
            delta_ratio = ((suggested_budget - target_amount) / target_amount) * 100
            delta_text = f"{delta_ratio:+.1f}%"
            delta_color = "inverse"
        else:
            delta_text = None
            delta_color = "inverse"

        st.metric(
            label="💡 Suggested Budget",
            value=format_currency(suggested_budget),
            delta=delta_text,
            delta_color=delta_color,
        )

    if ma_window == "weighted":
        ma_metric_label = "📊 Last 12 Months Weighted Avg"
        ma_metric_help = "Weighted average monthly expenses over the last 12 months (plot moving average window)"
    else:
        ma_metric_label = f"📊 Last {ma_window} Months Avg"
        ma_metric_help = f"Average monthly expenses over the last {ma_window} months (plot moving average window)"

    st.metric(
        label=ma_metric_label,
        value=format_currency(last_ma_avg),
        help=ma_metric_help,
    )


def _add_unified_plot_traces(
    fig, complete_monthly_data, plot_type, name, target_amount
):
    """Add bar, moving average, forecast, and target goal traces to unified plot"""
    expense_hover = [
        format_currency(abs(val)) for val in complete_monthly_data["amount"]
    ]

    fig.add_trace(
        go.Bar(
            x=complete_monthly_data["month"],
            y=abs(complete_monthly_data["amount"]),
            name="Actual Expenses",
            marker_color="#ff7f0e",
            opacity=0.8,
            hovertemplate="<b>%{x}</b><br>"
            + "Actual Expenses: %{customdata}<br>"
            + "<extra></extra>",
            customdata=expense_hover,
        )
    )

    ma_window = get_moving_average_window()
    ma_label = moving_average_label(ma_window)
    months_to_plot = complete_monthly_data["month"].tolist()
    full_series_abs = build_entity_monthly_series(
        plot_type, name, abs_values=True
    )
    ma_plot = moving_average_for_plot_months(
        full_series_abs, months_to_plot, window=ma_window
    )
    if ma_plot is not None:
        fig.add_trace(
            go.Scatter(
                x=months_to_plot,
                y=ma_plot,
                name=ma_label,
                line=dict(color="#1f77b4", width=2, dash="dot"),
                mode="lines",
                hovertemplate=f"<b>%{{x}}</b><br>{ma_label}: %{{customdata}}<br><extra></extra>",
                customdata=[
                    format_currency(v) if v is not None else "€0"
                    for v in ma_plot
                ],
            )
        )

    future_months = None
    if len(complete_monthly_data) >= 3:
        trend_line, forecast = calculate_forecast_trend(
            abs(complete_monthly_data["amount"])
        )
        trend_line_hover = [format_currency(val) for val in trend_line]

        fig.add_trace(
            go.Scatter(
                x=complete_monthly_data["month"],
                y=trend_line,
                name="12-Month Forecast Trend",
                line=dict(color="#d62728", width=2),
                mode="lines",
                hovertemplate="<b>%{x}</b><br>"
                + "12-Month Forecast Trend: %{customdata}<br>"
                + "<extra></extra>",
                customdata=trend_line_hover,
            )
        )

        last_month_date = complete_monthly_data["month_date"].iloc[-1]
        future_months = pd.date_range(
            start=last_month_date + pd.DateOffset(months=1),
            periods=3,
            freq="MS",
        )
        forecast_hover = [format_currency(val) for val in abs(forecast)]

        fig.add_trace(
            go.Scatter(
                x=future_months.strftime("%Y-%m"),
                y=abs(forecast),
                name="Forecast (Next 3 Months)",
                line=dict(color="#d62728", width=2, dash="dash"),
                mode="lines",
                hovertemplate="<b>%{x}</b><br>"
                + "Forecast: %{customdata}<br>"
                + "<extra></extra>",
                customdata=forecast_hover,
            )
        )

    if target_amount is not None and target_amount > 0:
        all_months = complete_monthly_data["month"].tolist()
        if future_months is not None:
            all_months.extend(future_months.strftime("%Y-%m").tolist())

        target_hover = [format_currency(target_amount)] * len(all_months)
        fig.add_trace(
            go.Scatter(
                x=all_months,
                y=[target_amount] * len(all_months),
                name=f"Target Goal ({format_currency(target_amount)})",
                line=dict(color="#2ca02c", width=3, dash="solid"),
                mode="lines",
                hovertemplate="<b>%{x}</b><br>"
                + "Target Goal: %{customdata}<br>"
                + "<extra></extra>",
                customdata=target_hover,
            )
        )


def create_unified_plot(
    name,
    transactions_df,
    budget_df,
    global_month_range,
    categories_data,
    plot_type="category",
    y_range=None,
):
    """
    Create comprehensive plot for either a category group or individual category

    Args:
        name: Category group name or category name
        transactions_df: DataFrame of transactions
        budget_df: DataFrame of budget data
        global_month_range: Global month range for consistent x-axis
        categories_data: List of category data dictionaries
        plot_type: 'category_group' or 'category'
    """
    prepared = _prepare_unified_plot_data(
        name, transactions_df, budget_df, global_month_range, categories_data, plot_type
    )
    if prepared is None:
        return None

    filtered_transactions, target_amount, avg_12_months, available_budget = prepared

    _render_unified_plot_metrics(
        name,
        transactions_df,
        global_month_range,
        plot_type,
        target_amount,
        avg_12_months,
        available_budget,
    )

    filtered_transactions["date"] = pd.to_datetime(filtered_transactions["date"])
    filtered_transactions["month"] = filtered_transactions["date"].dt.to_period("M")
    monthly_expenses = (
        filtered_transactions.groupby("month")["amount"].sum().reset_index()
    )
    monthly_expenses["month"] = monthly_expenses["month"].astype(str)

    fig = go.Figure()

    if not monthly_expenses.empty:
        monthly_expenses = monthly_expenses.sort_values("month")
        monthly_expenses["month_date"] = pd.to_datetime(monthly_expenses["month"])
        monthly_expenses = monthly_expenses.sort_values("month_date")

        all_months_df = pd.DataFrame(
            {
                "month_date": global_month_range,
                "month": global_month_range.strftime("%Y-%m"),
            }
        )
        monthly_expenses = monthly_expenses[
            monthly_expenses["month"].isin(all_months_df["month"])
        ]
        complete_monthly_data = (
            all_months_df.merge(
                monthly_expenses[["month", "amount"]], on="month", how="left"
            )
            .fillna(0)
            .sort_values("month_date")
        )

        _add_unified_plot_traces(
            fig, complete_monthly_data, plot_type, name, target_amount
        )

    layout_update = {
        "height": 400,
        "showlegend": False,
        "hovermode": "x unified",
        "xaxis_title": "Month",
        "yaxis_title": "Amount (€)",
        "barmode": "overlay",
    }

    if y_range is not None:
        layout_update["yaxis"] = {"range": y_range}

    fig.update_layout(**layout_update)

    return fig
