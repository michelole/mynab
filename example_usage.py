"""
Example usage of the YNAB API client
This script demonstrates basic usage of the official ynab library
"""

import ynab
import pandas as pd
from datetime import datetime, timedelta

def example_ynab_usage():
    """
    Example of how to use the YNAB API client
    Replace 'your_api_token_here' with your actual YNAB API token
    """
    
    # Your YNAB API token (get from https://app.youneedabudget.com/settings/developer)
    api_token = "your_api_token_here"
    
    try:
        # Configure the API client
        configuration = ynab.Configuration(access_token=api_token)
        
        with ynab.ApiClient(configuration) as api_client:
            # Get budgets
            budgets_api = ynab.BudgetsApi(api_client)
            budgets_response = budgets_api.get_budgets()
            
            if not budgets_response.data.budgets:
                print("No budgets found in your YNAB account.")
                return
            
            budget = budgets_response.data.budgets[0]
            print(f"Using budget: {budget.name}")
            
            # Get categories
            categories_api = ynab.CategoriesApi(api_client)
            categories_response = categories_api.get_categories(budget.id)
            
            print(f"Found {len(categories_response.data.category_groups)} category groups")
            
            # Get transactions for the last 6 months
            transactions_api = ynab.TransactionsApi(api_client)
            since_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            
            transactions_response = transactions_api.get_transactions(
                budget.id,
                since_date=since_date
            )
            
            print(f"Found {len(transactions_response.data.transactions)} transactions")
            
            # Process some basic data
            expenses = []
            for transaction in transactions_response.data.transactions:
                if transaction.amount < 0:  # Only expenses
                    expenses.append({
                        'date': transaction.date,
                        'amount': abs(transaction.amount) / 1000,  # Convert from millidollars
                        'category': transaction.category_name or "Uncategorized",
                        'payee': transaction.payee_name or "Unknown"
                    })
            
            if expenses:
                df = pd.DataFrame(expenses)
                print(f"\nTotal expenses: €{df['amount'].sum():,.2f}")
                print(f"Average daily expense: €{df['amount'].mean():,.2f}")
                
                # Group by category
                by_category = df.groupby('category')['amount'].sum().sort_values(ascending=False)
                print("\nTop 5 expense categories:")
                for category, amount in by_category.head().items():
                    print(f"  {category}: €{amount:,.2f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("YNAB API Example Usage")
    print("=" * 50)
    example_ynab_usage() 