# YNAB Dashboard Pages

This directory contains additional pages for the YNAB Budget Dashboard.

## Pages

### categories.py
- **Purpose**: Displays comprehensive analysis plots for all individual categories
- **Features**: 
  - Shows all category plots in a grid layout (2 columns)
  - Excludes Internal Master Category, Uncategorized, and Credit Card Payments
  - Each plot includes actual expenses, moving averages, forecast trends, and budget lines
  - No selectbox required - all categories are displayed automatically

## Navigation
- The main dashboard is in `../app.py`
- This page can be accessed via the sidebar navigation in Streamlit
- The page filename determines the order in the sidebar 