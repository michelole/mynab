"""Tests for the YNAB Budget Dashboard app."""

from collections.abc import Sequence
from datetime import date

import pandas as pd
import pytest


def test_calculate_moving_average() -> None:
    """Test moving average calculation."""
    from mynab.utils import calculate_moving_average

    # Test data
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # Test with window=3
    result = calculate_moving_average(data, window=3)
    expected = pd.Series([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])

    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_calculate_weighted_moving_average() -> None:
    """Test 12-month weighted moving average calculation."""
    from mynab.utils import (
        DEFAULT_MOVING_AVERAGE_WINDOW,
        MOVING_AVERAGE_WINDOW_OPTIONS,
        calculate_moving_average,
        calculate_series_tail_average,
        moving_average_label,
    )

    assert DEFAULT_MOVING_AVERAGE_WINDOW == "weighted"
    assert MOVING_AVERAGE_WINDOW_OPTIONS["12 months (weighted)"] == "weighted"
    assert moving_average_label("weighted") == "12-Month Weighted Moving Average"

    # 12 months: weights 1..12 for values 1..12 -> sum(i^2) / sum(i) = 650 / 78
    data = pd.Series(list(range(1, 13)), dtype=float)
    result = calculate_moving_average(data, window="weighted")
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx((1 * 1 + 2 * 2) / 3)
    assert result.iloc[-1] == pytest.approx(650.0 / 78.0)

    # Tail average helper matches last weighted moving average value
    assert calculate_series_tail_average(data, "weighted") == pytest.approx(
        650.0 / 78.0
    )


def test_calculate_forecast_trend() -> None:
    """Test forecast trend calculation."""
    from mynab.utils import calculate_forecast_trend

    # Test data
    data = pd.Series([1, 2, 3, 4, 5])

    # Test forecast
    trend_line, forecast = calculate_forecast_trend(data, periods=3)

    # Check that trend_line has same length as input
    assert len(trend_line) == len(data)

    # Check that forecast has expected length
    assert len(forecast) == 3

    # Check that trend_line is not all NaN
    assert not trend_line.isna().all()


def test_process_categories_data() -> None:
    """Test categories data processing."""
    from mynab.utils import process_categories_data

    # Mock categories response
    class MockCategory:
        def __init__(
            self,
            id: str,
            name: str,
            hidden: bool = False,
            deleted: bool = False,
            goal_target: int | None = None,
            goal_target_date: date | None = None,
        ) -> None:
            self.id = id
            self.name = name
            self.hidden = hidden
            self.deleted = deleted
            self.goal_target = goal_target
            self.goal_target_date = goal_target_date

    class MockCategoryGroup:
        def __init__(
            self, id: str, name: str, categories: Sequence[MockCategory]
        ) -> None:
            self.id = id
            self.name = name
            self.categories = categories

    class MockCategoriesData:
        def __init__(self, category_groups: Sequence[MockCategoryGroup]) -> None:
            self.category_groups = category_groups

    class MockCategoriesResponse:
        def __init__(self, category_groups: Sequence[MockCategoryGroup]) -> None:
            self.data = MockCategoriesData(category_groups)

    # Create mock data
    categories = [
        MockCategory(
            "cat1",
            "Food",
            False,
            False,
            goal_target=250000,
            goal_target_date=date(2026, 12, 1),
        ),
        MockCategory("cat2", "Transport", False, False),
        MockCategory("cat3", "Hidden", True, False),
        MockCategory("cat4", "Deleted", False, True),
    ]

    group = MockCategoryGroup("group1", "Living Expenses", categories)
    excluded_group = MockCategoryGroup(
        "group2", "Internal Master Category", [MockCategory("cat5", "Internal")]
    )
    response = MockCategoriesResponse([group, excluded_group])

    # Process data
    categories_data, category_groups = process_categories_data(response)

    # Check that hidden, deleted, and excluded-group categories are filtered out
    assert len(categories_data) == 2
    assert "Internal Master Category" not in category_groups
    assert any(
        cat["name"] == "Food"
        and cat.get("target_amount") == 250.0
        and cat.get("target_date") == date(2026, 12, 1)
        for cat in categories_data
    )
    assert any(cat["name"] == "Transport" for cat in categories_data)
    assert not any(cat["name"] == "Hidden" for cat in categories_data)
    assert not any(cat["name"] == "Deleted" for cat in categories_data)


def test_process_transactions_data() -> None:
    """Test transactions data processing including split subtransactions."""
    from mynab.utils import CategoryInfo, process_transactions_data

    class MockSubTransaction:
        def __init__(
            self,
            amount: int,
            category_id: str,
            category_name: str,
            memo: str | None = None,
        ) -> None:
            self.amount = amount
            self.category_id = category_id
            self.category_name = category_name
            self.memo = memo

    class MockTransaction:
        def __init__(
            self,
            date: str,
            amount: int,
            category_id: str | None = None,
            category_name: str | None = None,
            payee_name: str | None = None,
            memo: str | None = None,
            id: str | None = None,
            subtransactions: Sequence[MockSubTransaction] | None = None,
        ) -> None:
            self.date = date
            self.amount = amount
            self.category_id = category_id
            self.category_name = category_name
            self.payee_name = payee_name
            self.memo = memo
            self.id = id or "txn1"
            self.subtransactions = subtransactions

    class MockTransactionsData:
        def __init__(self, transactions: Sequence[MockTransaction]) -> None:
            self.transactions = transactions

    class MockTransactionsResponse:
        def __init__(self, transactions: Sequence[MockTransaction]) -> None:
            self.data = MockTransactionsData(transactions)

    transactions = [
        MockTransaction(
            "2024-01-01", -5000, "cat1", "Food", "Grocery Store", "Food shopping"
        ),
        MockTransaction("2024-01-02", 1000, "cat1", "Food", "Refund", "Return"),
        MockTransaction(
            "2024-01-03", -2000, "cat2", "Transport", "Gas Station", "Fuel"
        ),
        MockTransaction(
            "2024-01-04",
            -3000,
            payee_name="Supermarket",
            memo="Parent memo",
            id="txn_split",
            subtransactions=[
                MockSubTransaction(-1800, "cat1", "Food", "Split food"),
                MockSubTransaction(-1200, "cat2", "Transport"),
            ],
        ),
    ]

    response = MockTransactionsResponse(transactions)
    categories_data: list[CategoryInfo] = [
        {"id": "cat1", "name": "Food", "group": "Living Expenses"},
        {"id": "cat2", "name": "Transport", "group": "Living Expenses"},
    ]

    df = process_transactions_data(response, categories_data)

    assert len(df) == 5
    assert df.iloc[0]["amount"] == -5.0
    assert df.iloc[1]["amount"] == 1.0
    assert df.iloc[2]["amount"] == -2.0
    assert df.iloc[3]["amount"] == -1.8
    assert df.iloc[3]["memo"] == "Split food"
    assert df.iloc[4]["amount"] == -1.2
    assert df.iloc[4]["memo"] == "Parent memo"


def test_overview_monthly_totals_and_presets() -> None:
    """Test overview monthly aggregation, target delta formatting, and month range presets."""
    from mynab.utils import (
        compute_overview_monthly_totals,
        format_target_delta,
        resolve_month_range_preset,
    )

    df = pd.DataFrame(
        [
            {
                "date": "2025-01-10",
                "amount": 3000.0,
                "category": "Inflow: Ready to Assign",
                "category_group": "",
                "payee_name": "Employer",
            },
            {
                "date": "2025-01-15",
                "amount": -1200.0,
                "category": "Groceries",
                "category_group": "Necessidades",
                "payee_name": "Market",
            },
        ]
    )

    inc = compute_overview_monthly_totals(df, "total_income")
    exp = compute_overview_monthly_totals(df, "total_expense", abs_expense=True)
    net = compute_overview_monthly_totals(df, "total_net_income")

    assert inc.iloc[0] == pytest.approx(3000.0)
    assert exp.iloc[0] == pytest.approx(1200.0)
    assert net.iloc[0] == pytest.approx(1800.0)

    delta_text, delta_color = format_target_delta(110.7, 100.0)
    assert delta_text == "+11%"
    assert delta_color == "inverse"
    assert format_target_delta(100.0, 100.0, zero_text="On target") == (
        "On target",
        "inverse",
    )

    avail = [date(2025, m, 1) for m in range(1, 13)]
    assert resolve_month_range_preset("All Dates", avail) == (
        date(2025, 1, 1),
        date(2025, 12, 1),
    )


def test_multi_window_averages() -> None:
    """Test 3/6/12/24 month averages calculation across full history."""
    from mynab.utils import (
        AVERAGE_WINDOWS,
        calculate_entity_window_averages,
        calculate_series_window_averages,
    )

    assert AVERAGE_WINDOWS == (3, 6, 12, 24)

    series = pd.Series(list(range(1, 25)), dtype=float)
    avgs = calculate_series_window_averages(series)
    assert avgs[3] == pytest.approx(23.0)
    assert avgs[6] == pytest.approx(21.5)
    assert avgs[12] == pytest.approx(18.5)
    assert avgs[24] == pytest.approx(12.5)

    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    df = pd.DataFrame(
        [
            {
                "date": m.strftime("%Y-%m-%d"),
                "amount": -float(i + 1),
                "category": "Groceries",
                "category_group": "Necessidades",
            }
            for i, m in enumerate(months)
        ]
    )
    entity_avgs = calculate_entity_window_averages("category", "Groceries", df)
    assert entity_avgs == avgs


def test_ynab_sdk_models_integration() -> None:
    """Test processing functions against real ynab SDK models."""
    from uuid import uuid4

    from ynab.models import (
        Account,
        AccountsResponse,
        AccountsResponseData,
        AccountType,
        CategoriesResponse,
        CategoriesResponseData,
        Category,
        CategoryGroupWithCategories,
        MonthDetail,
        MonthDetailResponse,
        MonthDetailResponseData,
        TransactionClearedStatus,
        TransactionDetail,
        TransactionsResponse,
        TransactionsResponseData,
    )

    from mynab.utils import (
        _extract_accounts_list,
        process_categories_data,
        process_months_data,
        process_transactions_data,
    )

    group_id = uuid4()
    cat_id = uuid4()
    acc_id = uuid4()

    category = Category(
        id=cat_id,
        category_group_id=group_id,
        name="Groceries",
        hidden=False,
        internal=False,
        budgeted=300000,
        activity=-150000,
        balance=150000,
        goal_type="NEED",
        goal_target=300000,
        goal_target_date=date(2026, 10, 31),
        deleted=False,
    )
    group = CategoryGroupWithCategories(
        id=group_id,
        name="Living",
        hidden=False,
        internal=False,
        deleted=False,
        categories=[category],
    )
    categories_resp = CategoriesResponse(
        data=CategoriesResponseData(
            category_groups=[group],
            server_knowledge=1,
        )
    )
    categories_data, category_groups = process_categories_data(categories_resp)
    assert len(categories_data) == 1
    assert categories_data[0]["name"] == "Groceries"
    assert categories_data[0].get("target_amount") == 300.0
    assert categories_data[0].get("target_date") == date(2026, 10, 31)
    assert "Living" in category_groups

    tx = TransactionDetail(
        id="tx-1",
        var_date=date(2026, 10, 15),
        amount=-50000,
        cleared=TransactionClearedStatus.CLEARED,
        approved=True,
        account_id=acc_id,
        account_name="Checking",
        category_id=cat_id,
        category_name="Groceries",
        payee_name="Supermarket",
        deleted=False,
        subtransactions=[],
    )
    tx_resp = TransactionsResponse(
        data=TransactionsResponseData(transactions=[tx], server_knowledge=1)
    )
    tx_df = process_transactions_data(tx_resp, categories_data)
    assert len(tx_df) == 1
    assert tx_df.iloc[0]["amount"] == -50.0
    assert tx_df.iloc[0]["category_group"] == "Living"

    month_resp = MonthDetailResponse(
        data=MonthDetailResponseData(
            month=MonthDetail(
                month=date(2026, 10, 1),
                income=1000000,
                budgeted=300000,
                activity=-150000,
                to_be_budgeted=700000,
                deleted=False,
                categories=[category],
            )
        )
    )
    budget_df = process_months_data(month_resp, categories_data)
    assert len(budget_df) == 1
    assert budget_df.iloc[0]["budgeted"] == 300.0
    assert budget_df.iloc[0]["available"] == 150.0

    acc = Account(
        id=acc_id,
        name="Checking",
        type=AccountType.CHECKING,
        on_budget=True,
        closed=False,
        balance=500000,
        cleared_balance=500000,
        uncleared_balance=0,
        transfer_payee_id=uuid4(),
        deleted=False,
    )
    acc_resp = AccountsResponse(
        data=AccountsResponseData(accounts=[acc], server_knowledge=1)
    )
    acc_list = _extract_accounts_list(acc_resp)
    assert len(acc_list) == 1
    assert acc_list[0]["type"] == "checking"


if __name__ == "__main__":
    pytest.main([__file__])
