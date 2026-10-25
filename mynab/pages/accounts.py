import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from mynab.utils import (
    add_bar_trace,
    add_forecast_traces,
    add_moving_average_trace,
    apply_standard_plot_layout,
    build_account_net_worth_series,
    calculate_moving_average,
    calculate_series_window_averages,
    filter_data_by_date_range,
    format_currency,
    get_global_month_range,
    get_moving_average_window,
    map_account_net_worth_to_months,
    map_moving_average_to_months,
    render_average_metrics_grid,
    render_in_grid,
    render_page_header,
)


def _add_account_ma_and_forecast_traces(
    fig, complete_monthly_data, net_worth_full, months_to_plot, ma_window
):
    """Add moving average and forecast traces to an account plot if enough data exists."""
    if len(complete_monthly_data) < 3:
        return
    actual_data = complete_monthly_data[complete_monthly_data["net_worth"] != 0].copy()
    if len(actual_data) < 3:
        return

    ma_series = calculate_moving_average(abs(net_worth_full), window=ma_window)
    ma_plot = map_moving_average_to_months(ma_series, months_to_plot)
    add_moving_average_trace(fig, months_to_plot, ma_plot, ma_window=ma_window)

    add_forecast_traces(
        fig,
        actual_data["month"],
        actual_data["month_date"],
        abs(actual_data["net_worth"]),
        abs_forecast=True,
        trend_name="Forecast Trend",
    )


def create_account_plot(account_name, global_month_range):
    """Create a plot for a single account showing available, forecast, and multi-window averages"""
    end_period = pd.Period(global_month_range[-1].strftime("%Y-%m"), freq="M")
    net_worth_full = build_account_net_worth_series(account_name, end_period=end_period)
    if net_worth_full is None or net_worth_full.empty:
        return None

    months_to_plot = global_month_range.strftime("%Y-%m").tolist()
    complete_monthly_data = pd.DataFrame(
        {
            "month_date": global_month_range,
            "month": months_to_plot,
            "net_worth": map_account_net_worth_to_months(
                net_worth_full, months_to_plot
            ),
        }
    ).sort_values("month_date")

    averages = calculate_series_window_averages(net_worth_full)
    ma_window = get_moving_average_window()
    available = net_worth_full.iloc[-1]

    fig = go.Figure()
    add_bar_trace(
        fig,
        complete_monthly_data["month"],
        complete_monthly_data["net_worth"],
        "Net Worth",
        "#1f77b4",
    )
    _add_account_ma_and_forecast_traces(
        fig, complete_monthly_data, net_worth_full, months_to_plot, ma_window
    )
    apply_standard_plot_layout(fig, height=400, yaxis_title="Net Worth (€)")

    return fig, available, averages


def _render_account_card(account_name, global_month_range):
    """Render metrics and chart for a single account"""
    st.subheader(account_name)
    result = create_account_plot(account_name, global_month_range)
    if result is None:
        st.info(f"No data available for {account_name}")
        return

    fig, available, averages = result
    col1, _ = st.columns(2)
    with col1:
        st.metric("Available (Sum)", format_currency(available))
    render_average_metrics_grid(averages, short=True)

    st.plotly_chart(fig, width="stretch")


def _group_accounts_by_type(accounts_list):
    """Filter non-credit accounts and group them by account type"""
    accounts_by_type = {}
    for acc in accounts_list:
        if acc["type"] in ("credit", "creditCard"):
            continue
        acc_type = acc["type"] or "Other"
        accounts_by_type.setdefault(acc_type, []).append(acc)
    return accounts_by_type


def _render_account_type_sections(accounts_by_type, global_month_range):
    """Render each account type section and its grid of account cards."""
    for acc_type, accs in accounts_by_type.items():
        group_label = acc_type.title()
        if group_label == "Otherasset":
            group_label = "Tracking"
        st.markdown(f"---\n### {group_label} Accounts")
        render_in_grid(
            accs,
            lambda acc, mr=global_month_range: _render_account_card(acc["name"], mr),
        )


def main():
    render_page_header("🏦 Account Analysis")

    if not st.session_state.get("data_loaded", False):
        st.error("Data not loaded. Please go to the main page first.")
        return

    transactions_df = st.session_state.transactions_df
    accounts_list = st.session_state.get("accounts_list", [])
    if transactions_df is None or not accounts_list:
        st.error("Data not available. Please go to the main page first.")
        return

    accounts_by_type = _group_accounts_by_type(accounts_list)
    if not accounts_by_type:
        st.info("No non-credit account data available.")
        return

    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    filtered_transactions_df = filter_data_by_date_range(
        transactions_df, start_date, end_date
    )
    global_month_range, _, _ = get_global_month_range(
        filtered_transactions_df, start_date, end_date
    )
    _render_account_type_sections(accounts_by_type, global_month_range)


if __name__ == "__main__":
    main()
