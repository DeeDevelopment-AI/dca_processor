"""Main Streamlit dashboard application."""
import streamlit as st
import pandas as pd
from .components.filters import DashboardFilters
from .components.metrics import MetricsDisplay
from .components.charts import ChartCreator
from .components.user_activity import UserActivityDisplay
from .utils.styling import load_css
from db.queries import DatabaseQueries
from db.connection import DatabaseConnection
from config import load_config

class Dashboard:
    """Main dashboard class."""

    def __init__(self):
        """Initialize dashboard."""
        st.set_page_config(layout="wide", page_title="DCA Processor Dashboard")
        st.markdown(load_css(), unsafe_allow_html=True)

        # Initialize database connection and queries
        config = load_config()
        self.db_conn = DatabaseConnection(config.db)  # Your existing DatabaseConnection
        self.db_queries = DatabaseQueries(self.db_conn)

    def load_sample_data(self) -> dict:
        """Load sample data for the dashboard."""
        states_data = pd.DataFrame({
            'State': ['Arusha', 'Dar es Salaam', 'Dodoma', 'Iringa', 'Kigoma', 'Kilimanjaro', 'Mwanza'],
            'Investment': [30, 100, 220, 40, 90, 35, 30]
        })

        business_data = pd.DataFrame({
            'Business Type': ['Apartment', 'Farming', 'Office Bldg', 'Hospitality', 'Retail'],
            'Investment': [150, 120, 90, 40, 30]
        })

        regions_data = pd.DataFrame({
            'Region': ['Dodoma', 'Kigoma', 'Dar es Salaam', 'Mwanza'],
            'Percentage': [50, 18.7, 17.1, 14.2]
        })

        return {
            "states": states_data,
            "business": business_data,
            "regions": regions_data
        }

    def render_default_dashboard(self, data: dict):
        """Render the default dashboard view."""
        st.markdown("## DCA Processor Dashboard")

        # Workbook selector
        st.selectbox("📊 My Excel WorkBook", ["Workbook 1", "Workbook 2"])

        # Calculate and display metrics
        metrics_data = MetricsDisplay.calculate_metrics(data)
        MetricsDisplay.render_metrics_row(metrics_data)

        # Display charts
        ChartCreator.render_charts(data)

    def render_user_activity(self, filters: dict):
        """Render the user activity report."""
        st.markdown("## User Activity Report")
        st.markdown(f"Showing data from {filters['date_range']['start']} to {filters['date_range']['end']}")

        # Initialize and render user activity display with database queries
        user_activity = UserActivityDisplay(self.db_queries)
        user_activity.render(filters["date_range"])

    def run(self):
        """Run the dashboard application."""
        try:
            # Render sidebar filters
            filters = DashboardFilters.render_sidebar()

            # Handle different report types
            if filters["report"] == "Select Report":
                # Load and process default dashboard data
                data = self.load_sample_data()
                self.render_default_dashboard(data)

            elif filters["report"] == "Users Activity":
                self.render_user_activity(filters)

            elif filters["report"] in ["Token Activity (Global)", "Token Activity (Buy)", "Token Activity (Sell)"]:
                st.info(f"Coming soon: {filters['report']}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        finally:
            # Ensure database connection is closed properly
            if hasattr(self, 'db_conn'):
                self.db_conn.close()

def main():
    """Main entry point for the dashboard."""
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()