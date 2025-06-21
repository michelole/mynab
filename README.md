# YNAB Budget Dashboard

A comprehensive Streamlit dashboard for analyzing your YNAB (You Need A Budget) data with advanced visualizations and forecasting capabilities.

## Features

### 📊 Category Analysis
- **Per Category Plots**: Each category gets a comprehensive analysis including:
  - 12-month moving average trend line
  - 12-month forecast trend line
  - Monthly actual expenses (bar chart)
  - Monthly budget allocation (line chart)

### 📈 Group Analysis
- **Category Group Summary**: Stacked bar chart showing actual expenses per group
- **Group Trends**: Moving averages and forecast trends for each category group
- **Total Expense Trends**: Overall budget performance tracking

### 🔮 Forecasting
- Linear regression-based forecasting for 12-month trends
- Moving average calculations for trend analysis
- Visual comparison between actual spending and budgeted amounts

## Installation

### Prerequisites
- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd ynab
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create a virtual environment and install dependencies:
```bash
uv sync
```

4. Get your YNAB API token:
   - Go to [YNAB web app](https://app.youneedabudget.com/)
   - Click on your account name (top right)
   - Go to Account Settings
   - Click on Developer Settings
   - Copy your Personal Access Token

## Usage

### Quick Start
```bash
# Run the dashboard
uv run streamlit run app.py
```

### Alternative Methods
```bash
# Activate the virtual environment first
uv shell

# Then run the app
streamlit run app.py
```

Or use the installed script:
```bash
uv run mynab
```

### Development Setup
For development with additional tools:
```bash
uv sync --extra dev
```

## Data Sources

This app uses the official YNAB Python client to fetch:
- **Budgets**: Your budget information
- **Categories**: All categories and category groups
- **Transactions**: Last 24 months of transaction data
- **Months**: Budget allocation data by month

## Visualizations

### Category-Level Analysis
Each category displays:
- **Monthly Overview**: Bar chart of actual expenses with budget line overlay
- **Trends**: Moving average and forecast trend lines
- **Forecast Extension**: 12-month projection based on current trends

### Group-Level Analysis
- **Stacked Summary**: Monthly expenses broken down by category group
- **Trend Comparison**: Moving averages and forecasts across all groups
- **Total Performance**: Overall budget performance tracking

## Technical Details

- **API Client**: Uses the official `ynab` Python library (v1.4.0+)
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas for data manipulation
- **Caching**: Streamlit caching for API calls (1-hour TTL)
- **Forecasting**: Linear regression for trend prediction
- **Package Management**: uv for fast, reliable dependency management

## Requirements

- Python 3.8+
- YNAB account with API access
- Internet connection for API calls
- uv package manager

## Dependencies

### Core Dependencies
- `streamlit>=1.28.0`: Web app framework
- `ynab>=1.4.0`: Official YNAB API client
- `pandas>=2.0.0`: Data manipulation
- `plotly>=5.15.0`: Interactive visualizations
- `numpy>=1.24.0`: Numerical computations
- `python-dotenv>=1.0.0`: Environment variable management

### Development Dependencies
- `pytest>=7.0.0`: Testing framework
- `black>=23.0.0`: Code formatting
- `isort>=5.12.0`: Import sorting
- `flake8>=6.0.0`: Linting
- `mypy>=1.0.0`: Type checking

## Development

### Code Quality
```bash
# Format code
uv run black .

# Sort imports
uv run isort .

# Type checking
uv run mypy .

# Linting
uv run flake8 .
```

### Testing
```bash
uv run pytest
```

## License

This project is licensed under the Apache-2.0 license.

## Contributing

Feel free to submit issues and enhancement requests!