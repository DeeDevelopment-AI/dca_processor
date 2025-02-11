# update_tokens.py
from db.connection import DatabaseConnection
from db.queries import DatabaseQueries
from config import load_config
import logging
import sys
import time
from time import sleep


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('token_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def fetch_and_store_security_data(db_queries, api_key):
    """
    Fetch token security data from Birdeye API and store it in batches of 100.
    """
    try:
        logger.info("Fetching distinct tokens not already in `token_security`...")
        with db_queries.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT ti.token 
                FROM token_info ti
                LEFT JOIN token_security ts ON ti.token = ts.token
                WHERE ts.token IS NULL;
            """)
            tokens = [row['token'] for row in cursor.fetchall()]

        if not tokens:
            logger.warning("No new tokens found to process.")
            return

        logger.info(f"Found {len(tokens)} new tokens to process.")

        # Process tokens in batches of 100
        batch_size = 100
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}: {len(batch)} tokens")

            success = db_queries.fetch_and_store_token_security(batch, api_key)

            if success:
                logger.info(f"Batch {i // batch_size + 1} successfully inserted into DB.")
            else:
                logger.warning(f"Batch {i // batch_size + 1} had issues, check logs for details.")

            logger.info("Sleeping for 1 second to avoid DB overload...")
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error in fetch_and_store_security_data: {str(e)}", exc_info=True)


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
            sleep(1)
            # Update token information from DexTools API
            logger.info("Updating token info using DexTools v2 API...")
            db_queries.update_token_info_with_rate_limit(
                api_url=config.dextools.api_url,
                rate_limit=config.dextools.rate_limit,
                api_key=config.dextools.api_key
            )

            logger.info("Token information update completed successfully.")

            # Fetch and store security data from Birdeye API
            logger.info("Starting security data update process...")
            fetch_and_store_security_data(db_queries, config.birdeye.api_key)
            logger.info("Security data update completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred during processing: {str(e)}", exc_info=True)
            raise

        finally:
            logger.info("Closing database connection...")
            db_connection.close()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
