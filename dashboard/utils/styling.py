"""Module for dashboard styling and CSS."""

def load_css() -> str:
    """Return custom CSS styles for the dashboard."""
    return """
    <style>
        /* Dark theme */
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        
        /* Metric card styling */
        .metric-card {
            background-color: #2D3748;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #3e2c6a;
        }
        .css-1d391kg, div[class*="stSidebarNav"] {
            background-color: #171923;
        }
        
        /* Additional sidebar elements */
        .css-1wrcr25, .css-6qob1r, .css-1prua9e {
            background-color: #171923 !important;
        }
        
        /* Sidebar selection background */
        .stSelectbox [data-testid="stMarkdownContainer"] {
            background-color: #171923;
        }
        
        /* Multiselect background in sidebar */
        .stMultiSelect div[role="listbox"] {
            background-color: #171923;
        }
        
        /* Custom title */
        .custom-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
    </style>
    """

def apply_style_to_metric(label: str, value: str) -> str:
    """Create HTML for a styled metric card."""
    return f"""
    <div class="metric-card">
        <small>{label}</small>
        <h2>{value}</h2>
    </div>
    """