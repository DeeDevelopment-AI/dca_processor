# main.py
from db.connection import DatabaseConnection
from db.queries import DatabaseQueries
from config import load_config
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('token_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()

        # Initialize database connection and queries
        logger.info("Initializing database connection...")
        db_connection = DatabaseConnection(config.db)
        db_queries = DatabaseQueries(db_connection)

        try:
            # Call stored procedures if needed
            logger.info("Calling stored procedures...")
            db_queries.call_stored_procedures()

            # Update incomplete token records first
            logger.info("Updating incomplete token records...")
            db_queries.update_incomplete_token_info(
                api_url=config.dextools.api_url,
                rate_limit=config.dextools.rate_limit,
                api_key=config.dextools.api_key
            )

            logger.info(f"Updating token info using DexTools v2 API...")
            db_queries.update_token_info_with_rate_limit(
                api_url=config.dextools.api_url,
                rate_limit=config.dextools.rate_limit,
                api_key=config.dextools.api_key
            )

            logger.info("Token information update completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred during processing: {str(e)}")
            raise

        finally:
            logger.info("Closing database connection...")
            db_connection.close()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
