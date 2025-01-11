"""Module for dashboard chart components."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Any, Dict

class ChartCreator:
    """Class to manage chart creation and display."""

    @staticmethod
    def create_investment_line_chart(data: pd.DataFrame) -> go.Figure:
        """Create investment by state line chart."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['State'],
            y=data['Investment'],
            mode='lines',
            line=dict(color='#3182CE', width=2)
        ))
        fig.update_layout(
            title='Investment by State',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False
        )
        return fig

    @staticmethod
    def create_business_bar_chart(data: pd.DataFrame) -> go.Figure:
        """Create investment by business type bar chart."""
        fig = px.bar(
            data,
            x='Investment',
            y='Business Type',
            orientation='h'
        )
        fig.update_layout(
            title='Investment by Business Type',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        return fig

    @staticmethod
    def create_regions_pie_chart(data: pd.DataFrame) -> go.Figure:
        """Create regions by ratings pie chart."""
        fig = px.pie(
            data,
            values='Percentage',
            names='Region',
            title='Regions by Ratings'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        return fig

    @staticmethod
    def render_charts(data: Dict[str, pd.DataFrame]) -> None:
        """Render all charts in the dashboard."""
        chart_col1, chart_col2, chart_col3 = st.columns(3)

        with chart_col1:
            fig_line = ChartCreator.create_investment_line_chart(data["states"])
            st.plotly_chart(fig_line, use_container_width=True)

        with chart_col2:
            fig_bar = ChartCreator.create_business_bar_chart(data["business"])
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col3:
            fig_pie = ChartCreator.create_regions_pie_chart(data["regions"])
            st.plotly_chart(fig_pie, use_container_width=True)