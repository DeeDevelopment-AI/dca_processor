"""Module for dashboard metric components."""
import streamlit as st
from ..utils.styling import apply_style_to_metric
from typing import Dict, Any

class MetricsDisplay:
    """Class to manage metrics display."""

    @staticmethod
    def format_number(value: float) -> str:
        """Format numbers for display."""
        if value >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value/1_000:.2f}K"
        return f"{value:.2f}"

    @staticmethod
    def render_metrics_row(metrics_data: Dict[str, Any]) -> None:
        """Render a row of metric cards."""
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(
                apply_style_to_metric(
                    "🚀 Total Investment",
                    MetricsDisplay.format_number(metrics_data["total_investment"])
                ),
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                apply_style_to_metric(
                    "🔄 Most frequent",
                    MetricsDisplay.format_number(metrics_data["most_frequent"])
                ),
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                apply_style_to_metric(
                    "📈 Average",
                    MetricsDisplay.format_number(metrics_data["average"])
                ),
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                apply_style_to_metric(
                    "💰 Central Earnings",
                    MetricsDisplay.format_number(metrics_data["central_earnings"])
                ),
                unsafe_allow_html=True
            )

        with col5:
            st.markdown(
                apply_style_to_metric(
                    "⭐ Ratings",
                    MetricsDisplay.format_number(metrics_data["ratings"])
                ),
                unsafe_allow_html=True
            )

    @staticmethod
    def calculate_metrics(data: Any) -> Dict[str, float]:
        """Calculate metrics from data."""
        # Implement actual calculations based on your data structure
        return {
            "total_investment": 2_482_205_481,
            "most_frequent": 847_300,
            "average": 4_964_411,
            "central_earnings": 2_593_682,
            "ratings": 3_510
        }