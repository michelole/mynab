"""YNAB API fetching and raw response normalization."""

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from uuid import UUID

import pandas as pd
import streamlit as st
from ynab.api import AccountsApi, CategoriesApi, MonthsApi, PlansApi, TransactionsApi
from ynab.api_client import ApiClient
from ynab.configuration import Configuration
from ynab.exceptions import OpenApiException
from ynab.models import AccountsResponse, MonthDetailResponse

from mynab.config import settings
from mynab.models import (
    AccountInfo,
    BudgetRecord,
    CategoriesResponseLike,
    CategoryInfo,
    TransactionsResponseLike,
    YnabDataResult,
    _CategoryLike,
    _MonthCategoryLike,
    _TransactionLike,
    _TxItemLike,
    get_excluded_groups,
)


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


@st.cache_data(ttl=settings.cache_ttl)
def get_ynab_data(access_token: str) -> YnabDataResult:
    """Fetch data from YNAB API."""
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

            now = datetime.now(tz=UTC)
            start_date = now - timedelta(days=730)
            transactions_response = TransactionsApi(api_client).get_transactions(
                budget_id, since_date=start_date.date()
            )

            current_month = now.replace(day=1).date()
            months_response = MonthsApi(api_client).get_plan_month(budget_id, current_month)

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
        str(raw_goal_type) if isinstance(raw_goal_type, str) and raw_goal_type != "" else None
    )
    return {
        "id": category.id,
        "name": category.name,
        "group": group_name,
        "category_group_id": group_id,
        "target_amount": (
            (goal_target / 1000) if goal_target is not None and goal_target != 0 else None
        ),
        "target_type": target_type,
        "target_date": target_date,
    }


def process_categories_data(
    categories_response: CategoriesResponseLike,
) -> tuple[list[CategoryInfo], dict[str, list[CategoryInfo]]]:
    """Process categories data into a structured format, excluding specified groups."""
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
    """Find category group name for a given category ID."""
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
    """Extract transaction date from YNAB transaction object."""
    raw_date = getattr(transaction, "var_date", None) or getattr(transaction, "date", None)
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
    cat_info = cat_by_id.get(category_id) if category_id is not None and category_id != "" else None
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
    """Process transactions data into a structured format, handling split transactions."""
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
    """Process months data for budget information."""
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
