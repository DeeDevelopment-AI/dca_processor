from db.connection import DatabaseConnection
from db.queries import DatabaseQueries
from config import load_config
from models import DexToolsTokenDetails
from processors.update_tokens import main as update_tokens_main
import logging
import sys
import pandas as pd
from time import sleep

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("token_update.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def aggregate_token_data(transactions_df: pd.DataFrame, token_details_list: list[DexToolsTokenDetails]) -> pd.DataFrame:
    """
    Aggregate data for each distinct token using transaction data and detailed token information.

    Args:
        transactions_df (pd.DataFrame): DataFrame containing transaction data.
        token_details_list (List[DexToolsTokenDetails]): List of detailed token information.

    Returns:
        pd.DataFrame: Aggregated data for each token.
    """
    # Validate and filter token details list
    valid_details = [
        detail for detail in token_details_list if isinstance(detail, DexToolsTokenDetails)
    ]

    if not valid_details:
        logger.error("No valid DexToolsTokenDetails objects found in token_details_list.")
        raise ValueError("token_details_list is empty or invalid.")

    # Convert DexToolsTokenDetails objects to a DataFrame
    token_details_df = pd.DataFrame([detail.__dict__ for detail in valid_details])

    # Debugging: Ensure required columns exist in transactions_df
    required_columns = {"close_output_mint", "close_input_mint", "user_address", "out_amount"}
    if not required_columns.issubset(set(transactions_df.columns)):
        logger.error(f"Missing required columns in transactions_df: {required_columns - set(transactions_df.columns)}")
        raise ValueError("transactions_df is missing required columns for aggregation.")

    # Precompute matches for close_input_mint_count
    transactions_df["input_matches_output"] = transactions_df["close_input_mint"].isin(transactions_df["close_output_mint"])

    # Aggregate transaction data by close_output_mint
    transaction_aggregates = transactions_df.groupby("close_output_mint").agg(
        close_output_mint_count=("close_output_mint", "size"),
        close_input_mint_count=("input_matches_output", "sum"),
        distinct_users_close_output=("user_address", "nunique"),
        distinct_users_close_input=(
            "user_address",
            lambda x: transactions_df[transactions_df["close_input_mint"] == x.name]["user_address"].nunique(),
        ),
        total_out_amount=("out_amount", "sum"),
    ).reset_index()

    # Ensure total_out_amount is converted to float
    transaction_aggregates["total_out_amount"] = transaction_aggregates["total_out_amount"].astype(float)

    # Merge transaction aggregates with token details
    aggregated_data = pd.merge(
        transaction_aggregates,
        token_details_df,
        left_on="close_output_mint",
        right_on="token",
        how="left",
    )

    # Calculate total traded volume
    aggregated_data["total_traded_volume"] = (
            aggregated_data["total_out_amount"] * aggregated_data["price"]
    )

    # Select and rename columns for final output
    result = aggregated_data[
        [
            "token",
            "chain",
            "close_output_mint_count",
            "close_input_mint_count",
            "distinct_users_close_output",
            "distinct_users_close_input",
            "total_out_amount",
            "total_traded_volume",
            "price",
            "mcap",
            "holders",
            "price_5m",
            "variation_5m",
            "price_1h",
            "variation_1h",
            "price_6h",
            "variation_6h",
            "price_24h",
            "variation_24h",
        ]
    ].rename(
        columns={
            "token": "token_address",
            "price": "token_price",
            "mcap": "token_mcap",
            "holders": "token_holders",
        }
    )

    return result

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
            # Updating tokens
            logger.info("Calling update_tokens.main()...")
            update_tokens_main()

            # Get recent transactions
            logger.info("Fetching recent transactions...")
            transactions_df = db_queries.get_recent_transactions(n=2)

            # Extract unique token addresses
            logger.info("Extracting unique token addresses from transactions...")
            token_columns = [
                "open_input_mint",
                "open_output_mint",
                "close_input_mint",
                "close_output_mint",
            ]
            unique_tokens = pd.unique(transactions_df[token_columns].values.ravel())
            logger.info(f"Found {len(unique_tokens)} unique token addresses.")

            # Fetch detailed token information
            logger.info("Fetching detailed token information for each unique token...")
            detailed_token_info_list = []

            for token in unique_tokens:
                try:
                    details = db_queries.process_dextools_details(
                        chain="solana",
                        address=token,
                        api_url=config.dextools.api_url,
                        api_key=config.dextools.api_key,
                        rate_limit=1.0,
                    )
                    if details and isinstance(details, DexToolsTokenDetails):
                        detailed_token_info_list.append(details)
                    else:
                        logger.warning(f"No details found or invalid object for token: {token}")
                except Exception as e:
                    logger.error(f"Error fetching details for token {token}: {e}")
                # Respect DexTools rate limits
                sleep(1.0)

            # Aggregate token data
            logger.info("Aggregating token data...")
            aggregated_df = aggregate_token_data(transactions_df, detailed_token_info_list)

            # Save the aggregated results to an Excel file
            output_file = "aggregated_token_data.xlsx"
            logger.info(f"Saving aggregated results to {output_file}...")
            aggregated_df.to_excel(output_file, index=False)
            logger.info(f"Aggregated results successfully saved to {output_file}.")

        except Exception as e:
            logger.error(f"An error occurred during processing: {e}")
            raise

        finally:
            logger.info("Closing database connection...")
            db_connection.close()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
