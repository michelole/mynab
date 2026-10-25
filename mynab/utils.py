from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Literal, NotRequired, Protocol, TypedDict
from uuid import UUID

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from ynab.api import AccountsApi, CategoriesApi, MonthsApi, PlansApi, TransactionsApi
from ynab.api_client import ApiClient
from ynab.configuration import Configuration
from ynab.exceptions import OpenApiException
from ynab.models import (
    AccountsResponse,
    CategoriesResponse,
    MonthDetailResponse,
    TransactionsResponse,
)

import config

# Load environment variables
load_dotenv()

type MovingAverageWindow = int | Literal["weighted"]
type DeltaColor = Literal["normal", "inverse", "off"]
type DateLike = date | datetime | pd.Timestamp | str


class AccountInfo(TypedDict):
    id: str
    name: str
    type: str | None
    on_budget: bool | None
    closed: bool | None


class CategoryInfo(TypedDict):
    id: UUID | str
    name: str
    group: str
    category_group_id: NotRequired[UUID | str]
    target_amount: NotRequired[float | None]
    target_type: NotRequired[str | None]
    target_date: NotRequired[date | None]


class TransactionRecord(TypedDict):
    date: date
    amount: float
    category: str
    category_group: str
    payee_name: str
    memo: str
    transaction_id: str
    account_id: UUID | str | None
    account_name: str | None
    import_payee_name: str | None
    import_payee_name_original: str | None


class BudgetRecord(TypedDict):
    category: str
    category_group: str
    budgeted: float
    activity: float
    available: float
    target_amount: float | None
    target_type: str | None
    target_date: date | None


class _CategoryLike(Protocol):
    @property
    def id(self) -> UUID | str: ...
    @property
    def name(self) -> str: ...
    @property
    def hidden(self) -> bool: ...
    @property
    def deleted(self) -> bool: ...


class _CategoryGroupLike(Protocol):
    @property
    def id(self) -> UUID | str: ...
    @property
    def name(self) -> str: ...
    @property
    def categories(self) -> Sequence[_CategoryLike]: ...


class _CategoriesDataLike(Protocol):
    @property
    def category_groups(self) -> Sequence[_CategoryGroupLike]: ...


class CategoriesResponseLike(Protocol):
    @property
    def data(self) -> _CategoriesDataLike: ...


class _TxItemLike(Protocol):
    @property
    def amount(self) -> int | float: ...


class _TransactionLike(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def amount(self) -> int | float: ...


class _TransactionsDataLike(Protocol):
    @property
    def transactions(self) -> Sequence[_TransactionLike]: ...


class TransactionsResponseLike(Protocol):
    @property
    def data(self) -> _TransactionsDataLike: ...


class _MonthCategoryLike(Protocol):
    @property
    def id(self) -> UUID | str: ...
    @property
    def budgeted(self) -> int | float: ...
    @property
    def activity(self) -> int | float: ...
    @property
    def balance(self) -> int | float: ...


type YnabDataResult = (
    tuple[
        str,
        str,
        CategoriesResponse,
        TransactionsResponse,
        MonthDetailResponse,
        list[AccountInfo],
    ]
    | tuple[None, None, None, None, None, None]
)

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


def inject_custom_css() -> None:
    """Inject shared dashboard CSS styles into the current Streamlit page."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def render_page_header(title: str) -> None:
    """Render shared CSS and styled main header for a page."""
    inject_custom_css()
    st.markdown(f'<h1 class="main-header">{title}</h1>', unsafe_allow_html=True)


def render_in_grid[T](
    items: Sequence[T],
    render_item_fn: Callable[[T], None],
    cols_per_row: int = 2,
) -> None:
    """Render a list of items across a multi-column Streamlit grid."""
    for i in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, items[i : i + cols_per_row]):
            with col:
                render_item_fn(item)


def format_currency(amount: float | np.number | None) -> str:
    """Format currency amount with no cents and period as thousand separator"""
    if amount is None or pd.isna(amount):
        return "€0"

    rounded_amount = round(float(amount))
    if rounded_amount >= 0:
        return f"€{rounded_amount:,}".replace(",", ".")
    return f"-€{abs(rounded_amount):,}".replace(",", ".")


def _extract_accounts_list(accounts_response: AccountsResponse) -> list[AccountInfo]:
    """Convert YNAB accounts response into a list of dictionaries."""
    accounts = getattr(accounts_response.data, "accounts", None) or []
    result: list[AccountInfo] = []
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
def get_ynab_data(access_token: str) -> YnabDataResult:
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


def _build_category_dict(
    category: _CategoryLike, group_name: str, group_id: UUID | str
) -> CategoryInfo:
    """Extract a structured dictionary from a YNAB category model."""
    raw_target = getattr(category, "goal_target", None)
    goal_target = float(raw_target) if isinstance(raw_target, (int, float)) else None
    raw_date = getattr(category, "goal_target_date", None) or getattr(
        category, "goal_target_month", None
    )
    target_date = raw_date if isinstance(raw_date, date) else None
    raw_goal_type = getattr(category, "goal_type", None)
    target_type = (
        str(raw_goal_type)
        if isinstance(raw_goal_type, str) and raw_goal_type != ""
        else None
    )
    return {
        "id": category.id,
        "name": category.name,
        "group": group_name,
        "category_group_id": group_id,
        "target_amount": (
            (goal_target / 1000)
            if goal_target is not None and goal_target != 0
            else None
        ),
        "target_type": target_type,
        "target_date": target_date,
    }


def process_categories_data(
    categories_response: CategoriesResponseLike,
) -> tuple[list[CategoryInfo], dict[str, list[CategoryInfo]]]:
    """Process categories data into a structured format, excluding specified groups"""
    categories_data: list[CategoryInfo] = []
    category_groups: dict[str, list[CategoryInfo]] = {}
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


def _index_categories_by_id(
    categories_data: Sequence[CategoryInfo] | None,
) -> dict[UUID | str, CategoryInfo]:
    """Build a lookup dict mapping category ID to category metadata."""
    return {cat["id"]: cat for cat in (categories_data or []) if "id" in cat}


def _find_category_group(
    category_id: UUID | str | None,
    categories_data: Sequence[CategoryInfo] | Mapping[UUID | str, CategoryInfo],
) -> str:
    """Find category group name for a given category ID"""
    if category_id is None or category_id == "":
        return ""
    if isinstance(categories_data, Mapping):
        cat = categories_data.get(category_id)
        return str(cat["group"]) if cat is not None else ""
    for cat in categories_data:
        if cat["id"] == category_id:
            return str(cat["group"])
    return ""


def _get_transaction_date(transaction: _TransactionLike) -> date | str | None:
    """Extract transaction date from YNAB transaction object"""
    raw_date = getattr(transaction, "var_date", None) or getattr(
        transaction, "date", None
    )
    return raw_date if isinstance(raw_date, (date, str)) else None


def _build_tx_record(
    item: _TxItemLike,
    parent_tx: _TransactionLike,
    transaction_date: date | str,
    cat_by_id: Mapping[UUID | str, CategoryInfo],
) -> dict[str, object]:
    """Build a dictionary record for a transaction or split subtransaction."""
    raw_cat_id = getattr(item, "category_id", None)
    category_id = raw_cat_id if isinstance(raw_cat_id, (UUID, str)) else None
    cat_info = (
        cat_by_id.get(category_id)
        if category_id is not None and category_id != ""
        else None
    )
    category_group = cat_info["group"] if cat_info is not None else ""
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


def _extract_transaction_records(
    transaction: _TransactionLike,
    cat_by_id: Mapping[UUID | str, CategoryInfo],
) -> list[dict[str, object]]:
    """Extract one or more flat records from a YNAB transaction (including splits)."""
    if transaction.amount == 0:
        return []
    transaction_date = _get_transaction_date(transaction)
    if transaction_date is None:
        return []

    raw_subs = getattr(transaction, "subtransactions", None)
    if isinstance(raw_subs, Sequence) and len(raw_subs) > 0:
        return [
            _build_tx_record(sub, transaction, transaction_date, cat_by_id)
            for sub in raw_subs
            if sub.amount != 0
        ]
    return [_build_tx_record(transaction, transaction, transaction_date, cat_by_id)]


def process_transactions_data(
    transactions_response: TransactionsResponseLike,
    categories_data: Sequence[CategoryInfo] | None,
) -> pd.DataFrame:
    """Process transactions data into a structured format, handling split transactions"""
    cat_by_id = _index_categories_by_id(categories_data)
    transactions: list[dict[str, object]] = []
    for transaction in transactions_response.data.transactions:
        transactions.extend(_extract_transaction_records(transaction, cat_by_id))

    df = pd.DataFrame(transactions)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _build_month_category_record(
    category: _MonthCategoryLike,
    cat_by_id: Mapping[UUID | str, CategoryInfo],
) -> BudgetRecord:
    """Build a budget DataFrame record from a month category object."""
    cat_info = cat_by_id.get(category.id)
    return {
        "category": cat_info["name"] if cat_info else "Uncategorized",
        "category_group": cat_info["group"] if cat_info else "Uncategorized",
        "budgeted": category.budgeted / 1000,
        "activity": category.activity / 1000,
        "available": category.balance / 1000,
        "target_amount": cat_info.get("target_amount") if cat_info else None,
        "target_type": cat_info.get("target_type") if cat_info else None,
        "target_date": cat_info.get("target_date") if cat_info else None,
    }


def process_months_data(
    months_response: MonthDetailResponse | None,
    categories_data: Sequence[CategoryInfo] | None,
) -> pd.DataFrame:
    """Process months data for budget information"""
    if months_response is None:
        return pd.DataFrame()
    categories = months_response.data.month.categories or []
    cat_by_id = _index_categories_by_id(categories_data)

    budget_data = [
        _build_month_category_record(cat, cat_by_id)
        for cat in categories
        if cat.budgeted != 0 or cat.activity != 0
    ]
    return pd.DataFrame(budget_data)


MOVING_AVERAGE_WINDOW_OPTIONS: dict[str, MovingAverageWindow] = {
    "12 months (weighted)": "weighted",
    "3 months": 3,
    "6 months": 6,
    "12 months": 12,
    "24 months": 24,
}
DEFAULT_MOVING_AVERAGE_WINDOW: MovingAverageWindow = "weighted"
AVERAGE_WINDOWS: tuple[int, ...] = (3, 6, 12, 24)


def get_moving_average_window() -> MovingAverageWindow:
    """Moving average lookback from sidebar (plots only; forecast trend unchanged)."""
    window = st.session_state.get(
        "moving_average_window", DEFAULT_MOVING_AVERAGE_WINDOW
    )
    if window == "weighted":
        return "weighted"
    if isinstance(window, int) and window in MOVING_AVERAGE_WINDOW_OPTIONS.values():
        return window
    return DEFAULT_MOVING_AVERAGE_WINDOW


def moving_average_label(months: MovingAverageWindow) -> str:
    if months == "weighted":
        return "12-Month Weighted Moving Average"
    return f"{months}-Month Moving Average"


def map_moving_average_to_months(
    ma_series: pd.Series, months_to_plot: Sequence[str]
) -> list[float | None]:
    ma_map: dict[str, float] = {
        str(p): float(v) for p, v in zip(ma_series.index, ma_series.values)
    }
    return [ma_map.get(m, None) for m in months_to_plot]


def _group_monthly_amounts(df: pd.DataFrame) -> pd.Series:
    """Group a transactions DataFrame by month period and sum amounts."""
    tx = df.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    tx["month"] = tx["date"].dt.to_period("M")
    return pd.Series(tx.groupby("month")["amount"].sum()).sort_index()


def build_account_net_worth_series(
    account_name: str,
    transactions_df: pd.DataFrame | None = None,
    end_period: pd.Period | None = None,
) -> pd.Series | None:
    """Cumulative net worth by month from full account history."""
    transactions_df = _resolve_unfiltered_transactions(transactions_df)
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    txf = transactions_df[transactions_df["account_name"] == account_name]
    if txf.empty:
        return None

    monthly_amt = _group_monthly_amounts(txf)
    dataset_end_str = str(pd.to_datetime(transactions_df["date"]).max().to_period("M"))
    range_end_str = max(str(monthly_amt.index.max()), dataset_end_str)
    if end_period is not None:
        range_end_str = max(range_end_str, str(end_period))
    full_index = pd.period_range(
        start=str(monthly_amt.index.min()), end=range_end_str, freq="M"
    )
    return monthly_amt.reindex(full_index, fill_value=0).cumsum()


def map_account_net_worth_to_months(
    net_worth_series: pd.Series, months_to_plot: Sequence[str]
) -> list[float]:
    """Map cumulative net worth to plot months (0 before account activity, then ffill)."""
    nw_map: dict[str, float] = {
        str(p): float(v)
        for p, v in zip(net_worth_series.index, net_worth_series.values)
    }
    values: list[float] = []
    last_val = 0.0
    for month in months_to_plot:
        if month in nw_map:
            last_val = nw_map[month]
        values.append(last_val)
    return values


def _reindex_monthly_series(
    monthly: pd.Series | None,
    transactions_df: pd.DataFrame | None = None,
) -> pd.Series | None:
    if monthly is None or len(monthly) == 0:
        return None
    start_str = str(monthly.index.min())
    end_str = str(monthly.index.max())
    tx_df = (
        transactions_df
        if isinstance(transactions_df, pd.DataFrame) and not transactions_df.empty
        else _resolve_unfiltered_transactions()
    )
    full_index = get_full_month_period_index(tx_df)
    if full_index is not None and len(full_index) > 0:
        start_str = min(start_str, str(full_index.min()))
        end_str = max(end_str, str(full_index.max()))
    full_month_index = pd.period_range(
        start=start_str,
        end=end_str,
        freq="M",
    )
    return monthly.reindex(full_month_index, fill_value=0)


def build_entity_monthly_series(
    plot_type: str,
    name: str,
    transactions_df: pd.DataFrame | None = None,
    abs_values: bool = False,
) -> pd.Series | None:
    """Monthly totals from full transaction history (ignores sidebar date range)."""
    transactions_df = _resolve_unfiltered_transactions(transactions_df)
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    col = "category_group" if plot_type == "category_group" else "category"
    ft = transactions_df[transactions_df[col] == name]
    if ft.empty:
        return None

    series = _reindex_monthly_series(_group_monthly_amounts(ft), transactions_df)
    if series is None:
        return None
    return series.abs() if abs_values else series


def filter_income_transactions(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return inflow transactions excluding Starting Balance."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        df[
            (df["category"] == "Inflow: Ready to Assign")
            & (df["payee_name"] != "Starting Balance")
        ]
    ).copy()


def filter_categorized_expense_transactions(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return transactions that belong to a non-empty category group."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        df[
            (df["category_group"].astype(str) != "nan")
            & (df["category_group"].astype(str) != "")
        ]
    ).copy()


def resolve_selected_category_groups(
    selected_category_groups: list[str] | None = None,
) -> list[str]:
    """Resolve active category groups from session state or fallback to all non-excluded."""
    if selected_category_groups is None:
        state_groups = st.session_state.get("selected_category_groups")
        if isinstance(state_groups, list):
            selected_category_groups = [str(g) for g in state_groups]
    if isinstance(selected_category_groups, list) and len(selected_category_groups) > 0:
        return selected_category_groups
    excluded_groups = set(get_excluded_groups())
    category_groups = st.session_state.get("category_groups")
    if not isinstance(category_groups, Mapping):
        return []
    return sorted(str(g) for g in category_groups if str(g) not in excluded_groups)


def filter_transactions_by_selection(
    transactions_df: pd.DataFrame | None,
    selected_category_groups: Sequence[str],
    selected_categories: Sequence[str],
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
    preserve_income: bool = False,
) -> pd.DataFrame:
    """Filter transactions by excluded groups, optional date range, and selected categories."""
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return pd.DataFrame()

    excluded_groups = get_excluded_groups()
    df = pd.DataFrame(
        transactions_df[~transactions_df["category_group"].isin(excluded_groups)]
    ).copy()

    if start_date is not None and end_date is not None:
        df = filter_data_by_date_range(df, start_date, end_date)
    if df.empty:
        return df

    expense_df = pd.DataFrame(
        df[
            (df["category_group"].isin(selected_category_groups))
            & (df["category"].isin(selected_categories))
        ]
    )
    if not preserve_income:
        return expense_df.copy()

    income_df = filter_income_transactions(df)
    expense_only_df = pd.DataFrame(
        expense_df[expense_df["category"] != "Inflow: Ready to Assign"]
    )
    return pd.DataFrame(pd.concat([income_df, expense_only_df], ignore_index=True))


def filter_budget_by_selection(
    budget_df: pd.DataFrame | None,
    selected_category_groups: Sequence[str],
    selected_categories: Sequence[str],
) -> pd.DataFrame:
    """Filter budget DataFrame to selected category groups and categories."""
    if not isinstance(budget_df, pd.DataFrame) or budget_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        budget_df[
            (budget_df["category_group"].isin(selected_category_groups))
            & (budget_df["category"].isin(selected_categories))
        ]
    ).copy()


def filter_overview_transactions_for_ma(
    transactions_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Full history with overview sidebar category filters (no date filter)."""
    if transactions_df is None:
        state_df = st.session_state.get("transactions_df")
        transactions_df = state_df if isinstance(state_df, pd.DataFrame) else None
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return transactions_df

    selected_groups = resolve_selected_category_groups()
    state_cats = st.session_state.get("selected_categories")
    selected_categories: list[str] = (
        [str(c) for c in state_cats] if isinstance(state_cats, list) else []
    )
    return filter_transactions_by_selection(
        transactions_df,
        selected_groups,
        selected_categories,
        preserve_income=True,
    )


def _monthly_series_or_empty(sub_df: pd.DataFrame) -> pd.Series:
    """Group non-empty DataFrame by month or return an empty float Series."""
    return (
        _group_monthly_amounts(sub_df) if not sub_df.empty else pd.Series(dtype=float)
    )


def compute_overview_monthly_totals(
    df: pd.DataFrame | None, data_type: str, abs_expense: bool = False
) -> pd.Series:
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


def build_overview_monthly_series(
    data_type: str, transactions_df: pd.DataFrame | None = None
) -> pd.Series | None:
    """Monthly totals from full history for overview moving averages."""
    unfiltered_df = _resolve_unfiltered_transactions(transactions_df)
    if transactions_df is None:
        transactions_df = filter_overview_transactions_for_ma()
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    monthly = compute_overview_monthly_totals(
        transactions_df, data_type, abs_expense=(data_type == "total_expense")
    )
    if monthly.empty:
        return None
    return _reindex_monthly_series(monthly, unfiltered_df)


def _weighted_mean(values: pd.Series | np.ndarray) -> float:
    """Linearly weighted mean where the most recent month has the highest weight."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    weights = np.arange(1, len(arr) + 1, dtype=float)
    return float(np.average(arr, weights=weights))


def calculate_series_tail_average(
    series: pd.Series | None, window: MovingAverageWindow
) -> float:
    """Calculate absolute average of the tail of a monthly series for a given window."""
    if series is None or len(series) == 0:
        return 0.0
    if window == "weighted":
        return abs(_weighted_mean(series.tail(12)))
    return float(abs(series.tail(window).mean()))


def moving_average_for_plot_months(
    full_monthly_series: pd.Series | None,
    months_to_plot: Sequence[str],
    window: MovingAverageWindow | None = None,
) -> list[float | None] | None:
    """Moving average from full history, values for selected plot months only."""
    if full_monthly_series is None or len(full_monthly_series) == 0:
        return None
    if months_to_plot:
        full_index = pd.period_range(
            start=min(str(full_monthly_series.index.min()), min(months_to_plot)),
            end=max(str(full_monthly_series.index.max()), max(months_to_plot)),
            freq="M",
        )
        full_monthly_series = full_monthly_series.reindex(full_index, fill_value=0)
    if len(full_monthly_series) < 2:
        return None
    resolved_window = window if window is not None else get_moving_average_window()
    ma_full = calculate_moving_average(full_monthly_series, window=resolved_window)
    return map_moving_average_to_months(ma_full, months_to_plot)


def calculate_moving_average(
    data: pd.Series, window: MovingAverageWindow = 12
) -> pd.Series:
    """Calculate moving average for the given data"""
    if len(data) < 2:
        return data

    if window == "weighted":
        actual_window = min(12, len(data))
        return pd.Series(
            data.rolling(window=actual_window, min_periods=1).apply(
                _weighted_mean, raw=True
            ),
            index=data.index,
        )

    actual_window = min(window, len(data))
    return pd.Series(
        data.rolling(window=actual_window, min_periods=1).mean(),
        index=data.index,
    )


def calculate_forecast_trend(
    data: pd.Series, periods: int = config.FORECAST_PERIODS
) -> tuple[pd.Series, pd.Series]:
    """Calculate forecast trend using linear regression"""
    if len(data) < 2:
        return data, pd.Series(dtype=float)

    x = np.arange(len(data))
    coeffs = np.polyfit(x, data.to_numpy(dtype=float), 1)
    trend_line = np.polyval(coeffs, x)

    future_x = np.arange(len(data), len(data) + periods)
    forecast = np.polyval(coeffs, future_x)

    return pd.Series(trend_line, index=data.index), pd.Series(forecast, index=future_x)


def get_full_month_period_index(
    transactions_df: pd.DataFrame | None,
) -> pd.PeriodIndex | None:
    """Return a monthly PeriodIndex spanning the full date range of transactions_df."""
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None
    dates = pd.to_datetime(transactions_df["date"])
    return pd.period_range(
        start=dates.min().to_period("M"),
        end=dates.max().to_period("M"),
        freq="M",
    )


def calculate_series_window_averages(
    series: pd.Series | None, windows: Sequence[int] = AVERAGE_WINDOWS
) -> dict[int, float]:
    """Calculate tail averages for each window in windows (default 3/6/12/24 months)."""
    return {w: calculate_series_tail_average(series, w) for w in windows}


def _resolve_unfiltered_transactions(
    fallback_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Return full unfiltered transactions DataFrame from session state or fallback."""
    full_df = st.session_state.get("transactions_df")
    if isinstance(full_df, pd.DataFrame) and not full_df.empty:
        return full_df
    return fallback_df


def _resolve_entity_month_index(
    transactions_df: pd.DataFrame,
    global_month_range: pd.DatetimeIndex | None = None,
) -> pd.PeriodIndex | None:
    """Resolve the monthly PeriodIndex for reindexing entity expenses."""
    if global_month_range is not None and len(global_month_range) > 0:
        return pd.PeriodIndex(global_month_range.strftime("%Y-%m"), freq="M")
    return get_full_month_period_index(transactions_df)


def _build_entity_monthly_expenses(
    column: str,
    name: str,
    transactions_df: pd.DataFrame | None,
    global_month_range: pd.DatetimeIndex | None = None,
) -> pd.Series | None:
    """Build monthly expense series for a category or group, zero-filled across full months."""
    if not isinstance(transactions_df, pd.DataFrame) or transactions_df.empty:
        return None

    entity_tx = transactions_df[transactions_df[column] == name]
    monthly_expenses = _group_monthly_amounts(entity_tx)
    if monthly_expenses.empty:
        return None

    month_index = _resolve_entity_month_index(transactions_df, global_month_range)
    if month_index is not None and len(month_index) > 0:
        monthly_expenses = monthly_expenses.reindex(month_index, fill_value=0)
    return monthly_expenses


def _calculate_entity_averages(
    column: str,
    name: str,
    transactions_df: pd.DataFrame | None,
    months: MovingAverageWindow = 12,
    global_month_range: pd.DatetimeIndex | None = None,
) -> float:
    """Shared average spending calculation for a category or category group."""
    monthly_expenses = _build_entity_monthly_expenses(
        column, name, transactions_df, global_month_range
    )
    return calculate_series_tail_average(monthly_expenses, months)


def calculate_entity_window_averages(
    plot_type: str,
    name: str,
    transactions_df: pd.DataFrame | None = None,
    windows: Sequence[int] = AVERAGE_WINDOWS,
) -> dict[int, float]:
    """Calculate 3/6/12/24-month averages from full unfiltered transaction history."""
    tx_df = _resolve_unfiltered_transactions(transactions_df)
    column = "category_group" if plot_type == "category_group" else "category"
    monthly_expenses = _build_entity_monthly_expenses(column, name, tx_df)
    return calculate_series_window_averages(monthly_expenses, windows)


def calculate_category_averages(
    category_name: str,
    transactions_df: pd.DataFrame | None,
    months: MovingAverageWindow = 12,
    global_month_range: pd.DatetimeIndex | None = None,
) -> float:
    """Calculate average spending for a category over the last N months, including months with zero expenses"""
    return _calculate_entity_averages(
        "category", category_name, transactions_df, months, global_month_range
    )


def calculate_category_group_averages(
    group_name: str,
    transactions_df: pd.DataFrame | None,
    months: MovingAverageWindow = 3,
) -> float:
    """Calculate average spending for a category group over the last N months"""
    return _calculate_entity_averages(
        "category_group", group_name, transactions_df, months
    )


def calculate_category_available_budget(
    category_name: str, budget_df: pd.DataFrame | None
) -> float:
    """Calculate available budget for a category"""
    if not isinstance(budget_df, pd.DataFrame) or budget_df.empty:
        return 0.0
    cat_budget = budget_df[budget_df["category"] == category_name]
    return float(cat_budget["available"].iloc[0]) if not cat_budget.empty else 0.0


def calculate_category_group_available_budget(
    group_name: str, budget_df: pd.DataFrame | None
) -> float:
    """Calculate total available budget for a category group"""
    if not isinstance(budget_df, pd.DataFrame) or budget_df.empty:
        return 0.0
    group_budget = budget_df[budget_df["category_group"] == group_name]
    return float(group_budget["available"].sum()) if not group_budget.empty else 0.0


def month_start(d: DateLike) -> date:
    """First calendar day of the month containing d."""
    ts = pd.Timestamp(d)
    return date(int(ts.year), int(ts.month), 1)


def _subtract_months(month_date: date, months: int) -> date:
    """Return the first-of-month date `months` calendar months before `month_date`."""
    total_months = (month_date.year * 12 + (month_date.month - 1)) - months
    year, month_zero = divmod(total_months, 12)
    return date(year, month_zero + 1, 1)


def month_end(d: DateLike) -> date:
    """Last calendar day of the month containing d."""
    start_dt = month_start(d)
    ts = pd.Timestamp(
        start_dt.year, start_dt.month, start_dt.day
    ) + pd.offsets.MonthEnd(0)
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    return start_dt


def list_available_months(transactions_df: pd.DataFrame | None = None) -> list[date]:
    """First-of-month dates for every month in the data (or 2010 through today)."""
    today = date.today()
    if isinstance(transactions_df, pd.DataFrame) and not transactions_df.empty:
        dates = pd.to_datetime(transactions_df["date"])
        range_start = dates.min().replace(day=1)
        range_end = dates.max()
    else:
        range_start = pd.Timestamp(2010, 1, 1)
        range_end = pd.Timestamp(today.year, today.month, today.day)
    return [
        m.date() for m in pd.date_range(start=range_start, end=range_end, freq="MS")
    ]


def format_month_option(month_start_date: date) -> str:
    return month_start_date.strftime("%B %Y")


def dates_from_month_selection(
    start_month: DateLike, end_month: DateLike
) -> tuple[date, date]:
    """Inclusive bounds for a start month and end month (both first-of-month)."""
    return month_start(start_month), month_end(end_month)


def filter_data_by_date_range(
    transactions_df: pd.DataFrame, start_date: DateLike, end_date: DateLike
) -> pd.DataFrame:
    """Filter transactions to full months (start month through end month inclusive)."""
    if transactions_df.empty:
        return transactions_df

    start_dt = month_start(start_date)
    end_dt = month_end(end_date)
    range_start = pd.Timestamp(start_dt.year, start_dt.month, start_dt.day)
    range_end = pd.Timestamp(end_dt.year, end_dt.month, end_dt.day)

    filtered_df = transactions_df.copy()
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])

    mask = (filtered_df["date"] >= range_start) & (filtered_df["date"] <= range_end)
    return pd.DataFrame(filtered_df[mask])


def safe_strftime(dt: date | datetime | pd.Timestamp | str | None) -> str:
    """Safely format datetime objects for display"""
    try:
        if dt is None or pd.isna(dt):
            return "N/A"
        if isinstance(dt, (date, datetime, pd.Timestamp)):
            return dt.strftime("%Y-%m")
        return str(dt)
    except (ValueError, TypeError, AttributeError):
        return str(dt)


MONTH_RANGE_PRESET_OPTIONS: list[str] = [
    "Last 3 Months",
    "Last 6 Months",
    "Last 12 Months",
    "Year to Date",
    "Last Year",
    "All Dates",
    "Custom",
]
DEFAULT_MONTH_RANGE_PRESET: str = "Last 12 Months"

_PRESET_MONTH_OFFSETS: dict[str, int] = {
    "Last 3 Months": 2,
    "Last 6 Months": 5,
    "Last 12 Months": 11,
}


def get_default_end_month() -> date:
    """Latest month included in presets (current month from the 15th, else prior month)."""
    today = date.today()
    if today.day >= 15:
        return month_start(today)
    first_day_current_month = date(today.year, today.month, 1)
    return month_start(first_day_current_month - timedelta(days=1))


def clamp_month_range(
    start_month: date, end_month: date, available_months: Sequence[date]
) -> tuple[date, date]:
    if not available_months:
        return start_month, end_month
    clamped_start = max(month_start(start_month), available_months[0])
    clamped_end = min(month_start(end_month), available_months[-1])
    return clamped_start, max(clamped_start, clamped_end)


def _resolve_bounded_end_month(available_months: Sequence[date]) -> date:
    """Determine preset end month clamped to available months."""
    end_month = get_default_end_month()
    if available_months:
        end_month = min(month_start(end_month), available_months[-1])
        end_month = max(end_month, available_months[0])
    return end_month


def _compute_preset_unclamped_bounds(preset: str, end_month: date) -> tuple[date, date]:
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


def resolve_month_range_preset(
    preset: str, available_months: Sequence[date]
) -> tuple[date, date]:
    """Return (start_month, end_month) as first-of-month dates for a sidebar preset."""
    if preset == "All Dates":
        if available_months:
            return available_months[0], available_months[-1]
        end_month = get_default_end_month()
        return _subtract_months(end_month, 11), end_month

    end_month = _resolve_bounded_end_month(available_months)
    start_month, end_month = _compute_preset_unclamped_bounds(preset, end_month)
    return clamp_month_range(start_month, end_month, available_months)


def get_default_month_range() -> tuple[date, date]:
    """Default display range: last 12 calendar months (first-of-month for each bound)."""
    return resolve_month_range_preset("Last 12 Months", [])


def get_default_date_range() -> tuple[date, date]:
    """Backward-compatible inclusive date bounds from default month range."""
    start_month, end_month = get_default_month_range()
    return dates_from_month_selection(start_month, end_month)


def get_global_month_range(
    transactions_df: pd.DataFrame | None, start_date: DateLike, end_date: DateLike
) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    """Month range for plots, aligned to the sidebar month selection."""
    start_dt = month_start(start_date)
    end_dt = month_start(end_date)
    earliest_date = pd.Timestamp(start_dt.year, start_dt.month, start_dt.day)
    latest_date = pd.Timestamp(end_dt.year, end_dt.month, end_dt.day)
    assert isinstance(earliest_date, pd.Timestamp)
    assert isinstance(latest_date, pd.Timestamp)
    global_month_range = pd.date_range(start=earliest_date, end=latest_date, freq="MS")
    return global_month_range, earliest_date, latest_date


def get_excluded_groups() -> list[str]:
    """Get list of excluded category groups"""
    return [
        "Internal Master Category",
        "Uncategorized",
        "Credit Card Payments",
        "Hidden Categories",
    ]


def get_default_category_groups() -> list[str]:
    """Get default category groups for filtering"""
    return ["Lazer", "Necessidades"]


def get_default_categories() -> list[str]:
    """Get default categories for filtering"""
    return ["Groceries", "Dining Out", "Transportation"]


def _collect_entity_y_values(
    transactions_df: pd.DataFrame,
    categories_data: Sequence[CategoryInfo] | None,
    column_name: str,
    target_key: Literal["name", "group"],
    names: Sequence[str],
) -> list[float]:
    """Collect monthly expense and target y-values for categories or category groups"""
    y_values: list[float] = []
    cats = categories_data or []
    for name in names:
        filtered_tx = transactions_df[transactions_df[column_name] == name]
        if filtered_tx.empty:
            continue
        monthly_expenses = _group_monthly_amounts(filtered_tx)
        y_values.extend(float(v) for v in monthly_expenses.abs().values)
        for cat in cats:
            target_amt = cat.get("target_amount")
            if cat[target_key] == name and target_amt is not None:
                y_values.append(target_amt)
    return y_values


def _collect_overview_y_values(transactions_df: pd.DataFrame) -> list[float]:
    """Collect monthly income and expense y-values for overview plots"""
    inc = compute_overview_monthly_totals(transactions_df, "total_income")
    exp = compute_overview_monthly_totals(
        transactions_df, "total_expense", abs_expense=True
    )
    return [float(v) for v in inc.values] + [float(v) for v in exp.values]


def calculate_global_y_range(
    transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    categories_data: Sequence[CategoryInfo] | None,
    plot_type: str = "category",
    category_names: Sequence[str] | None = None,
) -> list[float] | None:
    """Calculate global y-axis range for a set of plots."""
    if transactions_df.empty:
        return None

    if plot_type == "overview":
        all_y_values = _collect_overview_y_values(transactions_df)
    elif (
        category_names is not None
        and len(category_names) > 0
        and plot_type in ("category", "category_group")
    ):
        target_key: Literal["name", "group"] = (
            "name" if plot_type == "category" else "group"
        )
        all_y_values = _collect_entity_y_values(
            transactions_df, categories_data, plot_type, target_key, category_names
        )
    else:
        all_y_values = []

    if not all_y_values:
        return None

    min_val, max_val = min(all_y_values), max(all_y_values)
    padding = (max_val - min_val) * 0.1
    return [max(0.0, min_val - padding), max_val + padding]


def format_target_delta(
    value: float,
    target_amount: float | None,
    zero_text: str | None = None,
) -> tuple[str | None, DeltaColor]:
    """Calculate percentage delta text and color against a target budget."""
    if target_amount is None or target_amount <= 0:
        return None, "normal"
    delta = value - target_amount
    if delta == 0 and zero_text is not None:
        return zero_text, "inverse"
    delta_pct = (delta / target_amount) * 100
    return f"{delta_pct:+.0f}%", "inverse"


def build_complete_monthly_df(
    global_month_range: pd.DatetimeIndex,
    monthly_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
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


def add_bar_trace(
    fig: go.Figure,
    x_months: Sequence[str] | pd.Series,
    y_values: Sequence[float] | pd.Series,
    name: str,
    color: str,
    hover_label: str | None = None,
) -> None:
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
            customdata=[format_currency(float(val)) for val in y_values],
        )
    )


def add_moving_average_trace(
    fig: go.Figure,
    months_to_plot: Sequence[str],
    ma_plot: Sequence[float | None] | None,
    ma_window: MovingAverageWindow | None = None,
) -> None:
    """Add a formatted moving average Scatter trace to a Plotly figure."""
    if ma_plot is None:
        return
    resolved_window = (
        ma_window if ma_window is not None else get_moving_average_window()
    )
    ma_label = moving_average_label(resolved_window)
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
    fig: go.Figure,
    months_series: Sequence[str] | pd.Series,
    month_dates_series: pd.Series,
    values_for_regression: pd.Series,
    abs_forecast: bool = True,
    trend_name: str = "12-Month Forecast Trend",
) -> pd.DatetimeIndex | None:
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
            customdata=[format_currency(float(val)) for val in trend_line],
        )
    )

    last_month_date = pd.Timestamp(month_dates_series.iloc[-1])
    future_months = pd.date_range(
        start=last_month_date + pd.DateOffset(months=1),
        periods=config.FORECAST_PERIODS,
        freq="MS",
    )
    forecast_values = forecast.abs() if abs_forecast else forecast
    fig.add_trace(
        go.Scatter(
            x=future_months.strftime("%Y-%m"),
            y=forecast_values,
            name="Forecast (Next 3 Months)",
            line={"color": "#d62728", "width": 2, "dash": "dash"},
            mode="lines",
            hovertemplate="<b>%{x}</b><br>Forecast: %{customdata}<br><extra></extra>",
            customdata=[format_currency(float(val)) for val in forecast_values],
        )
    )
    return future_months


def add_target_goal_trace(
    fig: go.Figure,
    base_months: Sequence[str],
    future_months: pd.DatetimeIndex | None,
    target_amount: float | None,
) -> None:
    """Add a horizontal target goal line across historical and forecast months."""
    if target_amount is None or target_amount <= 0:
        return
    all_months = list(base_months)
    if future_months is not None:
        all_months.extend(str(m) for m in future_months.strftime("%Y-%m"))
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


def apply_standard_plot_layout(
    fig: go.Figure,
    height: int = 400,
    y_range: Sequence[float] | None = None,
    yaxis_title: str = "Amount (€)",
) -> go.Figure:
    """Apply consistent layout settings to a Plotly figure."""
    layout_update: dict[str, object] = {
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


def render_average_metrics_grid(
    averages: Mapping[int, float],
    target_amount: float | None = None,
    short: bool = True,
) -> None:
    """Render a 2x2 grid of 3, 6, 12, and 24-month average metrics."""
    unit = "Avg" if short else "Average"
    for row_windows in ((3, 6), (12, 24)):
        cols = st.columns(2)
        for col, window in zip(cols, row_windows):
            with col:
                val = averages.get(window, 0.0)
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
    name: str,
    transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    categories_data: Sequence[CategoryInfo] | None,
    plot_type: str,
) -> tuple[pd.DataFrame, float | None, dict[int, float], float] | None:
    """Filter data and calculate summary metrics for unified plot"""
    cats = categories_data or []
    if plot_type == "category_group":
        filtered_transactions = pd.DataFrame(
            transactions_df[transactions_df["category_group"] == name]
        ).copy()
        target_amount: float | None = float(
            sum(
                amt
                for cat in cats
                if cat["group"] == name
                and (amt := cat.get("target_amount")) is not None
            )
        )
        available_budget = calculate_category_group_available_budget(name, budget_df)
    else:
        filtered_transactions = pd.DataFrame(
            transactions_df[transactions_df["category"] == name]
        ).copy()
        target_amount = next(
            (
                float(amt)
                for cat in cats
                if cat["name"] == name and (amt := cat.get("target_amount")) is not None
            ),
            None,
        )
        available_budget = calculate_category_available_budget(name, budget_df)

    if filtered_transactions.empty:
        return None

    averages = calculate_entity_window_averages(plot_type, name, transactions_df)
    return filtered_transactions, target_amount, averages, available_budget


def _render_unified_plot_metrics(
    target_amount: float | None,
    averages: Mapping[int, float],
    available_budget: float,
) -> None:
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
    fig: go.Figure,
    complete_monthly_data: pd.DataFrame,
    plot_type: str,
    name: str,
    target_amount: float | None,
    transactions_df: pd.DataFrame | None = None,
) -> None:
    """Add bar, moving average, forecast, and target goal traces to unified plot"""
    months_to_plot: list[str] = [
        str(m) for m in complete_monthly_data["month"].tolist()
    ]
    abs_amounts = complete_monthly_data["amount"].abs()

    add_bar_trace(fig, months_to_plot, abs_amounts, "Actual Expenses", "#ff7f0e")

    ma_window = get_moving_average_window()
    full_series_abs = build_entity_monthly_series(
        plot_type,
        name,
        transactions_df=_resolve_unfiltered_transactions(transactions_df),
        abs_values=True,
    )
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
    name: str,
    transactions_df: pd.DataFrame,
    budget_df: pd.DataFrame | None,
    global_month_range: pd.DatetimeIndex,
    categories_data: Sequence[CategoryInfo] | None,
    plot_type: str = "category",
    y_range: Sequence[float] | None = None,
) -> go.Figure | None:
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
            fig,
            complete_monthly_data,
            plot_type,
            name,
            target_amount,
            transactions_df=transactions_df,
        )

    return apply_standard_plot_layout(fig, height=400, y_range=y_range)
