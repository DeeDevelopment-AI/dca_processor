# main.py
import sys
import streamlit as st
from db.connection import DatabaseConnection
from db.queries import DatabaseQueries
from dashboard import run_dashboard
from config import load_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    try:
        # Load configuration
        config = load_config()

        # Setup database connection
        db_conn = DatabaseConnection(config.db)
        db_queries = DatabaseQueries(db_conn)

        # === LAUNCH THE STREAMLIT DASHBOARD ===
        run_dashboard(db_queries)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.exception(e)
        sys.exit(1)
    finally:
        if 'db_conn' in locals():
            db_conn.close()

if __name__ == "__main__":
    main()

