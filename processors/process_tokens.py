import os
import logging
import requests
import time
import pandas as pd
from tqdm import tqdm
from db.connection import DatabaseConnection
from db.queries import DatabaseQueries
from config import load_config
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
BIRDEYE_API_URL = "https://api.birdeye.so/public/price"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
RATE_LIMIT_DELAY = 1  # seconds
INTERVALS = [60, 240, 480, 1440]  # Minutes: 1h, 4h, 8h, 24h
PROCESSED_DIR = "./data/processed"

# Ensure processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def should_process_file(epoch_time):
    """Check if 24 hours have passed since the timestamp in the filename."""
    if epoch_time is None:
        return False

    current_time = int(datetime.now(timezone.utc).timestamp())

    if current_time >= epoch_time + (24 * 60 * 60):  # 24 hours have passed
        return True

    logger.info(f"Skipping file: epoch time {epoch_time} is not older than 24 hours (Current time: {current_time})")
    return False

def extract_epoch_from_filename(file_path):
    """Extract the epoch timestamp from the filename after the last underscore."""
    try:
        filename = os.path.basename(file_path)
        epoch_str = filename.split("_")[-1].replace(".csv", "")
        return int(epoch_str)
    except ValueError:
        logger.error(f"extract_epoch_from_filename: Invalid epoch format in filename '{file_path}'")
        return None

def fetch_historical_prices(token, timestamp, api_key):
    """Fetch historical prices at 5-minute intervals for the next 24 hours."""
    end_timestamp = timestamp + (24 * 60 * 60)  # 24 hours ahead
    url = f"https://public-api.birdeye.so/defi/history_price?address={token}&address_type=token&type=5m&time_from={timestamp}&time_to={end_timestamp}"
    logger.info(f"fetch_historical_price: Requesting history for token '{token}' from {timestamp} to {end_timestamp}.")
    headers = {
        "accept": "application/json",
        "x-chain": "solana",
        "X-API-KEY": api_key
    }
    for _ in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data.get("success") is True:
                items = data.get("data", {}).get("items", [])
                if items:
                    df = pd.DataFrame(items)
                    df = df.sort_values("unixTime")  # Ensure chronological order

                    logger.info(f"fetch_historical_price: Retrieved {len(df)} price snapshots for token '{token}'.")
                    logger.debug(f"fetch_historical_price: Data columns: {df.columns.tolist()}")
                    logger.debug(f"fetch_historical_price: Sample data:\n{df.head()}")

                    return df
                else:
                    logger.error(f"fetch_historical_price: No items found for token '{token}'. Response: {data}")
                    return None
            else:
                logger.error(f"fetch_historical_price: API returned error for token '{token}'. Response: {data}")
                return None

        except requests.RequestException as e:
            logger.error(f"fetch_historical_price: Request error for token '{token}': {e}")
            # Optionally uncomment the next line if you want to sleep between retries
            # time.sleep(RETRY_DELAY)

    logger.error(f"fetch_historical_price: Failed to fetch data for '{token}' after {MAX_RETRIES} retries.")
    return None

def extract_max_prices(df, start_time):
    """Extract max price ('value') at 1h, 4h, 8h, and 24h intervals after start_time."""
    if df is None or df.empty:
        logger.warning("extract_max_prices: No data available. Returning None for all intervals.")
        return [None] * len(INTERVALS)

    max_prices = []
    for minutes in INTERVALS:
        target_time = start_time + (minutes * 60)
        relevant_prices = df[df["unixTime"].between(start_time, target_time)]["value"]
        max_prices.append(relevant_prices.max() if not relevant_prices.empty else None)

    return max_prices

def calculate_percentage_changes(original_price, max_prices):
    """Calculate percentage increases from original price to max prices at different intervals.
    Returns 0 for any missing max_price instead of None.
    """
    if original_price is None or original_price <= 0:
        return [0] * len(max_prices)  # Avoid division by zero

    return [((max_price - original_price) / original_price * 100) if max_price is not None else 0 for max_price in max_prices]

def process_file(file_path, api_key):
    """Process a single file only if 24 hours have passed since its epoch timestamp."""
    epoch_time = extract_epoch_from_filename(file_path)

    # Ensure the file should be processed
    if not should_process_file(epoch_time):
        return

    try:
        df = pd.read_csv(file_path)

        # Ensure the file contains the 'price' column
        if "price" not in df.columns:
            logger.error(f"process_file: Missing 'price' column in {file_path}. Skipping file.")
            return

        updated_data = []

        logger.info(f"Processing file: {file_path} (Epoch: {epoch_time})")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(file_path)}"):
            token = row["token"]
            original_price = row["price"]  # Get the price from the CSV file
            logger.info(f"Fetching data for {token} at {epoch_time}...")

            # Fetch historical price data
            price_df = fetch_historical_prices(token, epoch_time, api_key)
            if price_df is None:
                logger.warning(f"Skipping {token} due to missing data.")
                continue

            # Extract max prices
            max_prices = extract_max_prices(price_df, epoch_time)

            # Calculate percentage increases (all values are now numeric)
            max_percent_changes = calculate_percentage_changes(original_price, max_prices)

            # Classification columns:
            # over_50: 1 if any percentage change is greater than 50, otherwise 0.
            over_50 = 1 if any(percent > 50 for percent in max_percent_changes) else 0

            # token_performance:
            # 'a' if any percentage change > 100,
            # 'b' if any percentage change between 50 and 100,
            # 'c' if any percentage change between 25 and 50,
            # otherwise 'd'
            if any(percent > 100 for percent in max_percent_changes):
                token_performance = 'a'
            elif any(50 < percent <= 100 for percent in max_percent_changes):
                token_performance = 'b'
            elif any(25 < percent <= 50 for percent in max_percent_changes):
                token_performance = 'c'
            else:
                token_performance = 'd'

            # Append results with original row, max prices, max percent changes, and classification columns.
            updated_data.append(list(row) + max_prices + max_percent_changes + [over_50, token_performance])

            # Respect API rate limits if necessary
            # time.sleep(RATE_LIMIT_DELAY)

        # Define new columns
        max_price_columns = ["max_price_1h", "max_price_4h", "max_price_8h", "max_price_24h"]
        max_percent_columns = ["max_percent_1h", "max_percent_4h", "max_percent_8h", "max_percent_24h"]
        classification_columns = ["over_50", "token_performance"]
        new_columns = max_price_columns + max_percent_columns + classification_columns

        # Create new DataFrame with additional columns
        df_new = pd.DataFrame(updated_data, columns=list(df.columns) + new_columns)

        # Save each file separately in the processed folder
        output_file = os.path.join(PROCESSED_DIR, os.path.basename(file_path))
        df_new.to_csv(output_file, index=False)

        logger.info(f"Saved processed data to {output_file}")

        # Remove the original file after successful processing
        os.remove(file_path)
        logger.info(f"Deleted original file: {file_path}")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)

def process_all_files(directory):
    """Process all CSV files in the given directory."""
    logger.info("Initializing database connection...")
    config = load_config()
    api_key = config.birdeye.api_key  # Fetch API key from config

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            process_file(file_path, api_key)  # Process each file independently

if __name__ == "__main__":
    input_directory = "./data/"  # Replace with the actual path if needed
    process_all_files(input_directory)
