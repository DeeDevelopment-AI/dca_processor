from db.connection import DatabaseConnection
from db.queries import DatabaseQueries
from processors.update_tokens import main as update_tokens_main
from models import TokenMetrics
from config import load_config
import requests
import logging
import sys
import time
from datetime import datetime
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('token_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def call_dextools_api(api_url, token, api_key, endpoint):
    """Call DexTools API for specific token information."""
    headers = {"X-API-KEY": api_key}
    url = f"{api_url}/v2/token/solana/{token}/{endpoint}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"DexTools API call failed for {token} at {endpoint}: {e}")
        return None

def process_tokens(recent_transactions, api_url, api_key):
    """Fetch and process distinct tokens from recent transactions."""
    # Extract distinct tokens from the recent transactions
    tokens = set(row['close_output_mint'] for row in recent_transactions if row['close_output_mint'])

    # Prepare results
    results = []

    for token in tokens:
        try:
            # Enforce rate limit: 1 API call per second
            time.sleep(1)

            # Fetch `info` data
            info_data = call_dextools_api(api_url, token, api_key, "info")
            if not info_data:
                continue

            # Fetch `price` data
            time.sleep(1)  # Enforce rate limit before next API call
            price_data = call_dextools_api(api_url, token, api_key, "price")
            if not price_data:
                continue

            # Aggregate metrics from recent transactions for this token
            output_count = sum(1 for row in recent_transactions if row['close_output_mint'] == token)
            distinct_output_users = len(set(row['user_address'] for row in recent_transactions if row['close_output_mint'] == token))
            total_out_amount = sum(Decimal(row['out_amount']) for row in recent_transactions if row['close_output_mint'] == token)

            token_price = Decimal(price_data.get("price", 0.0))
            token_metrics = TokenMetrics(
                address=token,
                name=info_data.get("name", "Unknown"),
                price=float(token_price),
                mcap=info_data.get("mcap", 0.0),
                holders=info_data.get("holders", 0),
                output_count=output_count,
                input_count=0,  # Placeholder as no input data is queried
                distinct_output_users=distinct_output_users,
                distinct_input_users=0,  # Placeholder as no input data is queried
                total_out_amount=float(total_out_amount),
                total_volume=float(total_out_amount * token_price),
                variation_5m=price_data.get("variation5m", 0.0),
                variation_1h=price_data.get("variation1h", 0.0),
                variation_6h=price_data.get("variation6h", 0.0),
                variation_24h=price_data.get("variation24h", 0.0)
            )
            results.append(token_metrics)
        except Exception as e:
            logger.error(f"Error processing token {token}: {e}")

    return results

def main():
    try:
        # Configuration
        logger.info("Loading configuration...")
        config = load_config()

        # Update tokens
        logger.info("Running update_tokens script...")
        update_tokens_main()

        # Initialize database connection and queries
        db_connection = DatabaseConnection(config.db)
        db_queries = DatabaseQueries(db_connection)

        try:
            # Call stored procedure and fetch recent transactions
            logger.info("Calling stored procedure `get_recent_transactions_hours`...")
            with db_queries.db.get_cursor() as cur:
                cur.execute("SELECT * FROM get_recent_transactions_hours(2);")
                recent_transactions = cur.fetchall()

            # Process tokens from recent transactions
            logger.info("Processing distinct tokens and fetching data...")
            results = process_tokens(recent_transactions, config.dextools.api_url, config.dextools.api_key)

            # Output results
            for result in results:
                logger.info(result)

        except Exception as e:
            logger.error(f"Error during database processing or token fetching: {e}")
        finally:
            db_connection.close()

    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
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
            logger.info("Requesting recent transactions...")
            recent_transactions_token = []
            recent_transactions_summary = db_queries.process_jupyter_dca(
                tokens = recent_transactions_token,
                api_url = config.dextools.api_url,
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
