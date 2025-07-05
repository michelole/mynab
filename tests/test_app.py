"""Tests for the YNAB Budget Dashboard app."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_calculate_moving_average():
    """Test moving average calculation."""
    from mynab.utils import calculate_moving_average
    
    # Test data
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    # Test with window=3
    result = calculate_moving_average(data, window=3)
    expected = pd.Series([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


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
        def __init__(self, id, name, hidden=False, deleted=False):
            self.id = id
            self.name = name
            self.hidden = hidden
            self.deleted = deleted
    
    class MockCategoryGroup:
        def __init__(self, id, name, categories):
            self.id = id
            self.name = name
            self.categories = categories
    
    class MockCategoriesResponse:
        def __init__(self, category_groups):
            self.data = type('Data', (), {'category_groups': category_groups})()
    
    # Create mock data
    categories = [
        MockCategory("cat1", "Food", False, False),
        MockCategory("cat2", "Transport", False, False),
        MockCategory("cat3", "Hidden", True, False),
        MockCategory("cat4", "Deleted", False, True),
    ]
    
    group = MockCategoryGroup("group1", "Living Expenses", categories)
    response = MockCategoriesResponse([group])
    
    # Process data
    categories_data, category_groups = process_categories_data(response)
    
    # Check that hidden and deleted categories are filtered out
    assert len(categories_data) == 2
    assert any(cat['name'] == 'Food' for cat in categories_data)
    assert any(cat['name'] == 'Transport' for cat in categories_data)
    assert not any(cat['name'] == 'Hidden' for cat in categories_data)
    assert not any(cat['name'] == 'Deleted' for cat in categories_data)


def test_process_transactions_data():
    """Test transactions data processing."""
    from mynab.utils import process_transactions_data
    
    # Mock transaction
    class MockTransaction:
        def __init__(self, date, amount, category_id=None, category_name=None, payee_name=None, memo=None, id=None):
            self.date = date
            self.amount = amount
            self.category_id = category_id
            self.category_name = category_name
            self.payee_name = payee_name
            self.memo = memo
            self.id = id or "txn1"
    
    class MockTransactionsResponse:
        def __init__(self, transactions):
            self.data = type('Data', (), {'transactions': transactions})()
    
    # Create mock data
    transactions = [
        MockTransaction("2024-01-01", -5000, "cat1", "Food", "Grocery Store", "Food shopping"),
        MockTransaction("2024-01-02", 1000, "cat1", "Food", "Refund", "Return"),  # Income
        MockTransaction("2024-01-03", -2000, "cat2", "Transport", "Gas Station", "Fuel"),
    ]
    
    response = MockTransactionsResponse(transactions)
    categories_data = [
        {'id': 'cat1', 'name': 'Food', 'group': 'Living Expenses'},
        {'id': 'cat2', 'name': 'Transport', 'group': 'Living Expenses'},
    ]
    
    # Process data
    df = process_transactions_data(response, categories_data)
    
    # Check that only expenses (negative amounts) are included
    assert len(df) == 2
    
    # Check that amounts are converted from millidollars
    assert df.iloc[0]['amount'] == 5.0  # -5000 millidollars = €5.00
    assert df.iloc[1]['amount'] == 2.0  # -2000 millidollars = €2.00


if __name__ == "__main__":
    pytest.main([__file__]) 