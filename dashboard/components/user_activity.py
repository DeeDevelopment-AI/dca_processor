# dashboard/components/user_activity.py
"""Module for user activity display component."""
import streamlit as st
from datetime import datetime
from typing import Dict, Any
import pandas as pd

class UserActivityDisplay:
    """Class to handle user activity display with pagination."""

    def __init__(self, db_queries):
        """Initialize with database queries instance."""
        self.db_queries = db_queries
        self.items_per_page = 25

    def render(self, date_range: Dict[str, datetime]):
        """Render the user activity component."""
        try:
            # Fetch user activity data using existing database queries
            users_data = self.db_queries.get_active_users_by_date_range(
                date_range["start"],
                date_range["end"]
            )

            if not users_data:
                st.warning("No user activity found for the selected date range.")
                return

            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(users_data)

            # Add index column starting from 1
            df.index = range(1, len(df) + 1)

            # Calculate total pages
            total_pages = len(df) // self.items_per_page + (1 if len(df) % self.items_per_page > 0 else 0)

            # Pagination controls
            col1, col2, col3 = st.columns([2, 3, 2])

            with col1:
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="user_activity_page"
                )

            with col2:
                st.markdown(f"### Showing {self.items_per_page} users per page")

            with col3:
                st.markdown(f"Total Pages: {total_pages}")

            # Calculate slice indices
            start_idx = (page - 1) * self.items_per_page
            end_idx = start_idx + self.items_per_page

            # Display the current page of data
            current_page_df = df.iloc[start_idx:end_idx].copy()

            # Format the user addresses to be more readable
            current_page_df['user_address'] = current_page_df['user_address'].apply(
                lambda x: f"{x[:6]}...{x[-4:]}"
            )

            # Rename columns for display
            current_page_df.columns = ['User Address', 'Transaction Count']

            # Display the table with custom styling
            st.markdown("""
                <style>
                    .dataframe {
                        font-size: 14px;
                        width: 100%;
                    }
                    .dataframe th {
                        background-color: #2D3748;
                        color: white;
                        text-align: left;
                        padding: 12px;
                    }
                    .dataframe td {
                        padding: 10px;
                        border-bottom: 1px solid #4A5568;
                    }
                    .dataframe tr:hover {
                        background-color: #2D3748;
                    }
                </style>
            """, unsafe_allow_html=True)

            st.table(current_page_df)

            # Display summary statistics
            st.markdown("### Summary Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Users", len(df))
            with col2:
                st.metric("Average Transactions",
                          f"{df['transaction_count'].mean():.2f}")
            with col3:
                st.metric("Max Transactions",
                          df['transaction_count'].max())

        except Exception as e:
            st.error(f"Error displaying user activity: {str(e)}")