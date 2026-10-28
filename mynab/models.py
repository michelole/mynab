"""Data models, Protocols, TypedDicts, and shared constants for the YNAB dashboard."""

from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal, NotRequired, Protocol, TypedDict
from uuid import UUID

import pandas as pd
from ynab.models import (
    CategoriesResponse,
    MonthDetailResponse,
    TransactionsResponse,
)

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

MOVING_AVERAGE_WINDOW_OPTIONS: dict[str, MovingAverageWindow] = {
    "12 months (weighted)": "weighted",
    "3 months": 3,
    "6 months": 6,
    "12 months": 12,
    "24 months": 24,
}
DEFAULT_MOVING_AVERAGE_WINDOW: MovingAverageWindow = "weighted"
AVERAGE_WINDOWS: tuple[int, ...] = (3, 6, 12, 24)
MIN_MONTHS_FOR_MA: int = 2
MIN_MONTHS_FOR_FORECAST: int = 3
MID_MONTH_CUTOFF_DAY: int = 15

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


def get_excluded_groups() -> list[str]:
    """Get list of excluded category groups."""
    return [
        "Internal Master Category",
        "Uncategorized",
        "Credit Card Payments",
        "Hidden Categories",
    ]


def get_default_category_groups() -> list[str]:
    """Get default category groups for filtering."""
    return ["Lazer", "Necessidades"]


def get_default_categories() -> list[str]:
    """Get default categories for filtering."""
    return ["Groceries", "Dining Out", "Transportation"]
