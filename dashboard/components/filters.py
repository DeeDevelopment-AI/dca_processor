"""Module for dashboard filter components."""
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any

class DashboardFilters:
    """Class to manage dashboard filters."""

    @staticmethod
    def render_sidebar() -> Dict[str, Any]:
        """Render sidebar filters and return selected values."""
        with st.sidebar:
            st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
            try:
                st.image("dashboard/assets/logo.png", width=150)
            except:
                st.markdown("### DCA Processor")
            st.markdown('</div>', unsafe_allow_html=True)

            # Report Selection
            st.markdown("### Select Report")
            report_options = [
                "Select Report",
                "Users Activity",
                "Token Activity (Global)",
                "Token Activity (Buy)",
                "Token Activity (Sell)"
            ]
            selected_report = st.selectbox(
                "Report Type",
                options=report_options,
                key="report_filter",
                label_visibility="collapsed"
            )

            # Date Range
            st.markdown("### Select Dates")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "From",
                    value=datetime.now() - timedelta(days=7),
                    key="date_from"
                )
            with col2:
                end_date = st.date_input(
                    "To",
                    value=datetime.now(),
                    key="date_to"
                )

            return {
                "report": selected_report,
                "date_range": {
                    "start": start_date,
                    "end": end_date
                }
            }

    @staticmethod
    def apply_filters(data: Any, filters: Dict[str, Any]) -> Any:
        """Apply selected filters to the data."""
        filtered_data = data

        # Filter by date range
        if "date_range" in filters:
            filtered_data = filtered_data[
                (filtered_data["date"] >= filters["date_range"]["start"]) &
                (filtered_data["date"] <= filters["date_range"]["end"])
                ]

        return filtered_data