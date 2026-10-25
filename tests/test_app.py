"""Tests for the YNAB Budget Dashboard app."""

from datetime import date
import pytest
import pandas as pd


def test_calculate_moving_average():
    """Test moving average calculation."""
    from mynab.utils import calculate_moving_average

    # Test data
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # Test with window=3
    result = calculate_moving_average(data, window=3)
    expected = pd.Series([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])

    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_calculate_weighted_moving_average():
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


def test_calculate_forecast_trend():
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


def test_process_categories_data():
    """Test categories data processing."""
    from mynab.utils import process_categories_data

    # Mock categories response
    class MockCategory:
        def __init__(self, id, name, hidden=False, deleted=False, goal_target=None):
            self.id = id
            self.name = name
            self.hidden = hidden
            self.deleted = deleted
            self.goal_target = goal_target

    class MockCategoryGroup:
        def __init__(self, id, name, categories):
            self.id = id
            self.name = name
            self.categories = categories

    class MockCategoriesResponse:
        def __init__(self, category_groups):
            self.data = type("Data", (), {"category_groups": category_groups})()

    # Create mock data
    categories = [
        MockCategory("cat1", "Food", False, False, goal_target=250000),
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
        cat["name"] == "Food" and cat["target_amount"] == 250.0
        for cat in categories_data
    )
    assert any(cat["name"] == "Transport" for cat in categories_data)
    assert not any(cat["name"] == "Hidden" for cat in categories_data)
    assert not any(cat["name"] == "Deleted" for cat in categories_data)


def test_process_transactions_data():
    """Test transactions data processing including split subtransactions."""
    from mynab.utils import process_transactions_data

    class MockSubTransaction:
        def __init__(self, amount, category_id, category_name, memo=None):
            self.amount = amount
            self.category_id = category_id
            self.category_name = category_name
            self.memo = memo

    class MockTransaction:
        def __init__(
            self,
            date,
            amount,
            category_id=None,
            category_name=None,
            payee_name=None,
            memo=None,
            id=None,
            subtransactions=None,
        ):
            self.date = date
            self.amount = amount
            self.category_id = category_id
            self.category_name = category_name
            self.payee_name = payee_name
            self.memo = memo
            self.id = id or "txn1"
            self.subtransactions = subtransactions

    class MockTransactionsResponse:
        def __init__(self, transactions):
            self.data = type("Data", (), {"transactions": transactions})()

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
    categories_data = [
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


def test_overview_monthly_totals_and_presets():
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

    delta_text, delta_color = format_target_delta(110.0, 100.0)
    assert delta_text == "+10.0%"
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


if __name__ == "__main__":
    pytest.main([__file__])
