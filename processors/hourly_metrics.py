from db.connection import DatabaseConnection
from db.queries import DatabaseQueries
from config import load_config
from tqdm import tqdm  # progress bar package
import logging
import sys
import pandas as pd
from time import sleep
import os
from datetime import datetime

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

def get_closest_price(token, block_time, token_history_data):
    """
    Given a token and a block_time, look up the historical price data for that token
    (as returned by fetch_historical_price) and return the price ('value') from the
    snapshot with the smallest time difference to block_time.
    """
    df = token_history_data.get(token)
    if df is None or df.empty:
        logger.warning(f"get_closest_price: No historical data available for token {token}.")
        return None
    # Calculate the absolute time difference between block_time and all snapshots
    time_diffs = (df["unixTime"] - block_time).abs()
    closest_idx = time_diffs.idxmin()
    closest_price = df.loc[closest_idx, "value"]
    return closest_price

# -------------------------------------------------------------------
def main():
    # Initialize configuration and database connections
    config = load_config()
    db_connection = DatabaseConnection(config.db)
    db_queries = DatabaseQueries(db_connection)

    try:
        logger.info("main: Starting main function execution.")

        # (1) Optionally update token information
        # logger.info("main: Calling update_tokens.main()...")
        # update_tokens_main()

        # (2) Fetch recent transactions (e.g., from the past 6 hours)
        logger.info("main: Fetching recent transactions...")
        transactions_df = db_queries.get_recent_transactions(n=2)
        logger.info(f"main: Fetched {len(transactions_df)} transactions.")

        # (3) Fetch token details once using the Birdeye API
        token_columns = ["open_input_mint", "open_output_mint", "close_input_mint", "close_output_mint"]
        unique_tokens = pd.unique(transactions_df[token_columns].values.ravel())
        logger.info(f"main: Found {len(unique_tokens)} unique token addresses for details.")
        chain_id = "solana"
        token_details_df = db_queries.fetch_token_details(chain_id, unique_tokens, config.birdeye.api_key)

        # (4) Add new columns for tokens and their historical prices
        transactions_df["input_token"] = None
        transactions_df["output_token"] = None
        transactions_df["input_price"] = None
        transactions_df["output_price"] = None

        # (5) For each distinct token, fetch its historical price snapshots once
        logger.info("main: Fetching historical price data for each unique token...")
        token_history_data = {}
        for token in tqdm(unique_tokens, desc="Fetching token historical prices", unit="token"):
            df_history = db_queries.fetch_historical_price(token, config.birdeye.api_key)
            if df_history is not None:
                token_history_data[token] = df_history
            else:
                logger.warning(f"main: No historical data found for token {token}.")
            #sleep(0.2)  # To avoid rate limiting

        # (6) Process each transaction to look up the closest historical price for both input and output tokens
        logger.info("main: Processing transactions to calculate historical prices...")
        for index, row in tqdm(transactions_df.iterrows(), total=len(transactions_df), desc="Processing transactions"):
            if pd.notnull(row["open_input_mint"]):
                input_token = row["open_input_mint"]
                output_token = row["open_output_mint"]
                block_time = row["open_block_time"]
            else:
                input_token = row["close_input_mint"]
                output_token = row["close_output_mint"]
                block_time = row["close_block_time"]

            transactions_df.at[index, "input_token"] = input_token
            transactions_df.at[index, "output_token"] = output_token

            # Look up the closest price snapshot for each token based on block_time
            input_price = get_closest_price(input_token, block_time, token_history_data)
            output_price = get_closest_price(output_token, block_time, token_history_data)

            transactions_df.at[index, "input_price"] = input_price
            transactions_df.at[index, "output_price"] = output_price

        # (7) Calculate per-transaction dollar amounts
        logger.info("main: Calculating per-transaction dollar values...")
        transactions_df["dollars_sold"] = (
                transactions_df["in_amount"].astype(float) *
                transactions_df["input_price"].astype(float)
        )
        transactions_df["dollars_bought"] = (
                transactions_df["out_amount"].astype(float) *
                transactions_df["output_price"].astype(float)
        )

        # (8) Aggregate dollar amounts per token
        logger.info("main: Aggregating dollar values per token...")
        aggregated_sold = (
            transactions_df.groupby("input_token")["dollars_sold"]
            .sum()
            .reset_index()
            .rename(columns={"input_token": "token"})
        )
        aggregated_bought = (
            transactions_df.groupby("output_token")["dollars_bought"]
            .sum()
            .reset_index()
            .rename(columns={"output_token": "token"})
        )
        aggregated = pd.merge(aggregated_sold, aggregated_bought, on="token", how="outer")
        aggregated["dollars_sold"] = aggregated["dollars_sold"].fillna(0)
        aggregated["dollars_bought"] = aggregated["dollars_bought"].fillna(0)
        logger.info("main: Aggregated token values:")
        logger.info(aggregated)

        # (9) Merge aggregated data with token details
        # We merge on the token address field ("address") from the Birdeye API data.
        logger.info("main: Merging aggregated data with token details...")
        final_df = pd.merge(aggregated, token_details_df, left_on="token", right_on="address", how="left")

        # (10) Format dollars columns to use a comma as the decimal separator
        #logger.info("main: Formatting dollar values with comma as decimal separator...")
        #final_df["dollars_sold"] = final_df["dollars_sold"].apply(lambda x: f"{x:.2f}".replace('.',','))
        #final_df["dollars_bought"] = final_df["dollars_bought"].apply(lambda x: f"{x:.2f}".replace('.',','))

        # (11) Remove timezone information for Excel compatibility
        for col in transactions_df.select_dtypes(include=["datetimetz"]).columns:
            transactions_df[col] = transactions_df[col].dt.tz_localize(None)
        for col in final_df.select_dtypes(include=["datetimetz"]).columns:
            final_df[col] = final_df[col].dt.tz_localize(None)

        # (12) Create a folder to store generated Excel files (if not exists) and write the final merged output to an Excel file.
        output_folder = "./data/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"main: Created output folder '{output_folder}'.")

        # Generate Excel filename using epoch time
        timestamp = int(datetime.now().timestamp())  # Convert current time to epoch
        csv_filename = f"final_aggregated_token_details_{timestamp}.csv"
        output_path = os.path.join(output_folder, csv_filename)

        logger.info(f"main: Writing final aggregated token details to Excel file at '{output_path}'...")
        final_df.to_csv(output_path, index=False)
        logger.info("main: Final CSV file written successfully.")
        print(final_df)


    except Exception as e:
        logger.error(f"main: An error occurred: {e}", exc_info=True)
    finally:
        db_connection.close()
        logger.info("main: Database connection closed.")

if __name__ == "__main__":
    main()