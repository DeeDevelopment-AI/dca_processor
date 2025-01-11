# dashboard.py

import streamlit as st
import plotly.express as px
from datetime import date
from db.queries import DatabaseQueries

def show_most_out_tokens_per_day(db_queries: DatabaseQueries):
    st.subheader("Most Output Mint Tokens Per Day (Excluding Suspicious)")
    col1, col2 = st.columns(2)
    with col1:
        selected_start = st.date_input("Start Date", value=date(2024, 1, 1))
    with col2:
        selected_end = st.date_input("End Date", value=date(2024, 12, 31))

    df = db_queries.get_tokens_most_out_per_day(
        start_date=selected_start,
        end_date=selected_end
    )

    st.write("Columns found:", df.columns.tolist())
    st.dataframe(df.head())

    if df.empty:
        st.write("No data found for the selected date range.")
        return

    # Now we plot color='token_name'
    fig = px.bar(
        df,
        x="day",
        y="total_out",
        color="token_name",  # Must exist in df.columns
        title="Daily Output Amount by Token Name"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_top10_tokens_by_outflow(db_queries: DatabaseQueries):
    st.subheader("Top 10 Tokens by Outflow (Excluding Suspicious)")
    df = db_queries.get_top10_tokens_by_outflow()
    if df.empty:
        st.write("No tokens found.")
        return

    st.dataframe(df)
    fig = px.bar(
        df,
        x="token_name",
        y="total_out",
        title="Top 10 Tokens by Outflow"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_tokens_out_more_than_in(db_queries: DatabaseQueries):
    st.subheader("Token Trading Volumes Over Time")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", value=date(2024, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=date(2024, 12, 31))
    with col3:
        top_n = st.number_input("Top N Tokens", min_value=1, max_value=20, value=10)

    df = db_queries.get_tokens_out_more_than_in(start_date, end_date, top_n)

    if df.empty:
        st.write("No data found for the selected date range.")
        return

    fig = px.bar(
        df,
        x='date',
        y='volume',
        color='token_name',
        title=f"Daily Trading Volume by Token (Top {top_n})",
        labels={'volume': 'Volume', 'date': 'Date', 'token_name': 'Token'},
        barmode='stack'
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
        legend_title="Tokens",
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def show_daily_transaction_volume(db_queries: DatabaseQueries):
    st.subheader("Daily Transaction Volume (Excluding Suspicious)")
    df = db_queries.get_daily_transaction_volume()
    if df.empty:
        st.write("No volume data found.")
        return

    st.dataframe(df)
    fig = px.line(df, x="day", y="daily_volume", title="Daily Volume Over Time")
    st.plotly_chart(fig, use_container_width=True)

def show_most_active_users(db_queries: DatabaseQueries):
    st.subheader("Most Active User Addresses (Excluding Suspicious Tokens)")
    df = db_queries.get_most_active_users(limit=10)
    if df.empty:
        st.write("No user data found.")
        return

    st.dataframe(df)

def show_all_solana_dca_data(db_queries: DatabaseQueries):
    st.subheader("All Rows (Joined with token_info, Excluding Suspicious)")
    df = db_queries.load_solana_dca_data()
    st.write(f"Total rows: {len(df)}")
    st.dataframe(df.head(50))

def run_dashboard(db_queries: DatabaseQueries):
    st.title("Solana DCA Detailed Dashboard (Token Info + Non-Suspicious)")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select a metric to explore",
        [
            "Most Output Mint Tokens Per Day",
            "Top 10 Tokens by Outflow",
            "Tokens Out > In",
            "Daily Transaction Volume",
            "Most Active Users",
            "All DCA Data"
        ]
    )

    if page == "Most Output Mint Tokens Per Day":
        show_most_out_tokens_per_day(db_queries)
    elif page == "Top 10 Tokens by Outflow":
        show_top10_tokens_by_outflow(db_queries)
    elif page == "Tokens Out > In":
        show_tokens_out_more_than_in(db_queries)
    elif page == "Daily Transaction Volume":
        show_daily_transaction_volume(db_queries)
    elif page == "Most Active Users":
        show_most_active_users(db_queries)
    else:
        show_all_solana_dca_data(db_queries)

def main():
    st.warning("Replace this stub with actual DB connection logic, then call run_dashboard(db_queries).")
    # from db.connection import DatabaseConnection
    # from config import load_config
    # config = load_config()
    # db_conn = DatabaseConnection(config.db)
    # db_queries = DatabaseQueries(db_conn)
    # run_dashboard(db_queries)

if __name__ == "__main__":
    main()
