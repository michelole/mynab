import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from mynab.utils import (
    filter_data_by_date_range,
    get_global_month_range,
    calculate_moving_average,
    calculate_forecast_trend,
    format_currency,
    get_moving_average_window,
    moving_average_label,
    map_moving_average_to_months,
    build_account_net_worth_series,
    map_account_net_worth_to_months,
)

# Page configuration
st.set_page_config(
    page_title="Account Analysis - YNAB Dashboard",
    page_icon="🏦",
    layout="wide",
)

# Custom CSS for better styling
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


def create_account_plot(account_name, global_month_range):
    """Create a plot for a single account showing available, forecast, and 12-months average"""
    end_period = pd.Period(global_month_range[-1].strftime("%Y-%m"), freq="M")
    net_worth_full = build_account_net_worth_series(
        account_name, end_period=end_period
    )
    if net_worth_full is None or net_worth_full.empty:
        return None

    all_months_df = pd.DataFrame(
        {
            "month_date": global_month_range,
            "month": global_month_range.strftime("%Y-%m"),
        }
    )
    months_to_plot = all_months_df["month"].tolist()
    net_worth_plot = map_account_net_worth_to_months(net_worth_full, months_to_plot)
    complete_monthly_data = all_months_df.copy()
    complete_monthly_data["net_worth"] = net_worth_plot
    complete_monthly_data = complete_monthly_data.sort_values("month_date")

    avg_12_months = abs(net_worth_full.tail(12).mean())
    ma_window = get_moving_average_window()
    metric_ma_avg = abs(net_worth_full.tail(ma_window).mean())
    available = net_worth_full.iloc[-1]

    full_series = abs(net_worth_full)
    ma_series = None
    trend_line = None
    forecast = None
    future_months = None
    actual_data = None

    if len(complete_monthly_data) >= 3:
        actual_data_mask = complete_monthly_data["net_worth"] != 0
        actual_data = complete_monthly_data[actual_data_mask].copy()
        if len(actual_data) >= 3:
            ma_series = calculate_moving_average(full_series, window=ma_window)
            trend_line, forecast = calculate_forecast_trend(
                abs(actual_data["net_worth"])
            )
            last_month_date = actual_data["month_date"].iloc[-1]
            future_months = pd.date_range(
                start=last_month_date + pd.DateOffset(months=1),
                periods=3,
                freq="MS",
            )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=complete_monthly_data["month"],
            y=complete_monthly_data["net_worth"],
            name="Net Worth",
            marker_color="#1f77b4",
            opacity=0.8,
            hovertemplate="<b>%{x}</b><br>Net Worth: %{customdata}<br><extra></extra>",
            customdata=[
                format_currency(val) for val in complete_monthly_data["net_worth"]
            ],
        )
    )
    if ma_series is not None:
        ma_plot = map_moving_average_to_months(ma_series, months_to_plot)
        ma_label = moving_average_label(ma_window)
        fig.add_trace(
            go.Scatter(
                x=months_to_plot,
                y=ma_plot,
                name=ma_label,
                line=dict(color="#1f77b4", width=2, dash="dot"),
                mode="lines",
                hovertemplate=f"<b>%{{x}}</b><br>{ma_label}: %{{customdata}}<br><extra></extra>",
                customdata=[
                    format_currency(v) if v is not None else "€0" for v in ma_plot
                ],
            )
        )
    if trend_line is not None and actual_data is not None:
        fig.add_trace(
            go.Scatter(
                x=actual_data["month"],
                y=trend_line,
                name="Forecast Trend",
                line=dict(color="#d62728", width=2),
                mode="lines",
                hovertemplate="<b>%{x}</b><br>Forecast Trend: %{customdata}<br><extra></extra>",
                customdata=[format_currency(val) for val in trend_line],
            )
        )
    if forecast is not None and future_months is not None:
        fig.add_trace(
            go.Scatter(
                x=future_months.strftime("%Y-%m"),
                y=abs(forecast),
                name="Forecast (Next 3 Months)",
                line=dict(color="#d62728", width=2, dash="dash"),
                mode="lines",
                hovertemplate="<b>%{x}</b><br>Forecast: %{customdata}<br><extra></extra>",
                customdata=[format_currency(val) for val in abs(forecast)],
            )
        )
    fig.update_layout(
        height=400,
        showlegend=False,
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Net Worth (€)",
        barmode="overlay",
    )
    return fig, available, avg_12_months, metric_ma_avg, ma_window


def main():
    st.markdown(
        '<h1 class="main-header">🏦 Account Analysis</h1>',
        unsafe_allow_html=True,
    )

    if not st.session_state.get("data_loaded", False):
        st.error("Data not loaded. Please go to the main page first.")
        return

    transactions_df = st.session_state.transactions_df
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    accounts_list = st.session_state.get("accounts_list", [])

    if transactions_df is None or not accounts_list:
        st.error("Data not available. Please go to the main page first.")
        return

    non_credit_accounts = [
        acc for acc in accounts_list if acc["type"] not in ("credit", "creditCard")
    ]
    if not non_credit_accounts:
        st.info("No non-credit account data available.")
        return

    accounts_by_type = {}
    for acc in non_credit_accounts:
        acc_type = acc["type"] or "Other"
        if acc_type not in accounts_by_type:
            accounts_by_type[acc_type] = []
        accounts_by_type[acc_type].append(acc)

    filtered_transactions_df = filter_data_by_date_range(
        transactions_df, start_date, end_date
    )

    global_month_range, _, _ = get_global_month_range(
        filtered_transactions_df, start_date, end_date
    )

    for acc_type, accs in accounts_by_type.items():
        group_label = acc_type.title()
        if group_label == "Otherasset":
            group_label = "Tracking"
        st.markdown(f"---\n### {group_label} Accounts")
        cols_per_row = 2
        for i in range(0, len(accs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(accs):
                    acc = accs[i + j]
                    account_name = acc["name"]
                    with col:
                        st.subheader(account_name)
                        fig, available, avg_12_months, metric_ma_avg, ma_window = (
                            create_account_plot(account_name, global_month_range)
                        )
                        mcol1, mcol2, mcol3 = st.columns(3)
                        with mcol1:
                            st.metric("Available (Sum)", format_currency(available))
                        with mcol2:
                            st.metric("12-Months Avg", format_currency(avg_12_months))
                        with mcol3:
                            st.metric(
                                f"{ma_window}-Months Avg",
                                format_currency(metric_ma_avg),
                            )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No data available for {account_name}")


if __name__ == "__main__":
    main()
