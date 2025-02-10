from typing import List, Optional, Dict, Any
from models import BlockGap, DCATransaction, DexToolsTokenInfo, DexToolsTokenDetails
from db.connection import DatabaseConnection
import time
from time import sleep
from tqdm import tqdm  # progress bar package
import requests
import logging
import pandas as pd
from collections import defaultdict
from datetime import datetime, date
import json

import re

CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')


def strip_control_chars(text: str) -> str:
    return CONTROL_CHAR_RE.sub('', text)


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DatabaseQueries:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    def get_unique_mints(self, table_name: str = 'jupiter_dca_transactions') -> List[str]:
        """Get unique mints from all mint fields that are not already stored in token_info"""
        try:
            logger.info(f"Starting get_unique_mints for table: {table_name}")

            with self.db.get_cursor() as cur:
                logger.debug("Executing SQL query to fetch unique mints...")

                query = """
                    SELECT DISTINCT mint
                    FROM (
                        SELECT open_input_mint as mint FROM jupiter_dca_transactions
                        UNION
                        SELECT open_output_mint as mint FROM jupiter_dca_transactions
                        UNION
                        SELECT close_input_mint as mint FROM jupiter_dca_transactions
                        UNION
                        SELECT close_output_mint as mint FROM jupiter_dca_transactions
                    ) all_mints
                    WHERE mint NOT IN (SELECT token FROM token_info)
                    AND mint IS NOT NULL
                """

                logger.debug(f"Query: {query}")
                cur.execute(query)

                logger.debug("Fetching results...")
                results = cur.fetchall()
                logger.debug(f"Raw results: {results[:5]}...")  # Log first 5 results

                # Changed this line to access by column name 'mint' instead of index
                unique_mints = [row['mint'] for row in results]
                logger.info(f"Found {len(unique_mints)} unique mints across all mint fields in {table_name}")
                logger.debug(f"First few mints: {unique_mints[:5]}...")

                return unique_mints

        except Exception as e:
            logger.error(f"Error in get_unique_mints: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error traceback: ", exc_info=True)
            raise

    def store_token_info(self, tokens: List[DexToolsTokenInfo]):
        """
        Upsert the list of DexToolsTokenInfo into token_info table.
        """
        if not tokens:
            return False

        with self.db.get_cursor() as cursor:
            for t in tokens:
                # Convert to JSON for social_info, buy_tax, sell_tax
                social_info_json = None
                buy_tax_json = None
                sell_tax_json = None

                if t.social_info:
                    social_info_json = json.dumps(t.social_info)
                if t.buy_tax:
                    buy_tax_json = json.dumps(t.buy_tax)
                if t.sell_tax:
                    sell_tax_json = json.dumps(t.sell_tax)

                query = """
                    INSERT INTO token_info (
                        token, 
                        name, 
                        symbol,
                        logo,
                        description,
                        creation_time,
                        creation_block,
                        decimals,
                        social_info,
                        is_open_source,
                        is_honeypot,
                        is_mintable,
                        is_proxy,
                        slippage_modifiable,
                        is_blacklisted,
                        sell_tax,
                        buy_tax,
                        is_contract_renounced,
                        is_potentially_scam,
                        updated_at,
                        is_suspicious
                    )
                    VALUES (
                        %(token)s,
                        %(name)s,
                        %(symbol)s,
                        %(logo)s,
                        %(description)s,
                        %(creation_time)s,
                        %(creation_block)s,
                        %(decimals)s,
                        %(social_info)s,
                        %(is_open_source)s,
                        %(is_honeypot)s,
                        %(is_mintable)s,
                        %(is_proxy)s,
                        %(slippage_modifiable)s,
                        %(is_blacklisted)s,
                        %(sell_tax)s,
                        %(buy_tax)s,
                        %(is_contract_renounced)s,
                        %(is_potentially_scam)s,
                        %(updated_at)s,
                        %(is_suspicious)s
                    )
                    ON CONFLICT (token)
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        symbol = EXCLUDED.symbol,
                        logo = EXCLUDED.logo,
                        description = EXCLUDED.description,
                        creation_time = EXCLUDED.creation_time,
                        creation_block = EXCLUDED.creation_block,
                        decimals = EXCLUDED.decimals,
                        social_info = EXCLUDED.social_info,
                        is_open_source = EXCLUDED.is_open_source,
                        is_honeypot = EXCLUDED.is_honeypot,
                        is_mintable = EXCLUDED.is_mintable,
                        is_proxy = EXCLUDED.is_proxy,
                        slippage_modifiable = EXCLUDED.slippage_modifiable,
                        is_blacklisted = EXCLUDED.is_blacklisted,
                        sell_tax = EXCLUDED.sell_tax,
                        buy_tax = EXCLUDED.buy_tax,
                        is_contract_renounced = EXCLUDED.is_contract_renounced,
                        is_potentially_scam = EXCLUDED.is_potentially_scam,
                        updated_at = EXCLUDED.updated_at,
                        is_suspicious = EXCLUDED.is_suspicious,
                        last_checked_at = NOW();
                """

                params = {
                    "token": t.token,
                    "name": t.name,
                    "symbol": t.symbol,
                    "logo": t.logo,
                    "description": t.description,
                    "creation_time": t.creation_time,
                    "creation_block": t.creation_block,
                    "decimals": t.decimals,

                    "social_info": social_info_json,
                    "is_open_source": t.is_open_source,
                    "is_honeypot": t.is_honeypot,
                    "is_mintable": t.is_mintable,
                    "is_proxy": t.is_proxy,
                    "slippage_modifiable": t.slippage_modifiable,
                    "is_blacklisted": t.is_blacklisted,
                    "sell_tax": sell_tax_json,
                    "buy_tax": buy_tax_json,
                    "is_contract_renounced": t.is_contract_renounced,
                    "is_potentially_scam": t.is_potentially_scam,
                    "updated_at": t.updated_at,
                    "is_suspicious": t.is_suspicious
                }

                cursor.execute(query, params)

        return True

    def process_dextools_response(self, info_data: dict, audit_data: dict):
        """Processes API responses to extract token info."""
        base = info_data.get("data", {}) or {}
        audit = audit_data.get("data", {}) or {}

        # Parse base fields
        token_address = strip_control_chars(base.get("address", "") or "")
        token_name = strip_control_chars(base.get("name", "") or "")
        token_symbol = strip_control_chars(base.get("symbol", "") or "")
        token_logo = base.get("logo")
        token_desc = base.get("description")
        creation_time = base.get("creationTime")
        creation_block = base.get("creationBlock")
        decimals = base.get("decimals")
        social_info = base.get("socialInfo", {})

        # Convert creation_time to datetime
        dt_creation = None
        if creation_time:
            try:
                dt_creation = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
            except Exception:
                logger.warning(f"Invalid creationTime format: {creation_time}")

        # Parse audit fields
        is_open_source = audit.get("isOpenSource")
        is_honeypot = audit.get("isHoneypot")
        is_mintable = audit.get("isMintable")
        is_proxy = audit.get("isProxy")
        slippage_modifiable = audit.get("slippageModifiable")
        is_blacklisted = audit.get("isBlacklisted")
        sell_tax = audit.get("sellTax", {})
        buy_tax = audit.get("buyTax", {})
        is_contract_renounced = audit.get("isContractRenounced")
        is_potentially_scam = audit.get("isPotentiallyScam")
        updated_at_str = audit.get("updatedAt")

        dt_updated = None
        if updated_at_str:
            try:
                dt_updated = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
            except Exception:
                logger.warning(f"Invalid updatedAt format: {updated_at_str}")

        # Determine if token is suspicious
        suspicious_flag = is_potentially_scam == "yes" or is_blacklisted == "yes"

        return DexToolsTokenInfo(
            token=token_address,
            name=token_name,
            symbol=token_symbol,
            logo=token_logo,
            description=token_desc,
            creation_time=dt_creation,
            creation_block=creation_block,
            decimals=decimals,
            social_info=social_info,
            is_open_source=is_open_source,
            is_honeypot=is_honeypot,
            is_mintable=is_mintable,
            is_proxy=is_proxy,
            slippage_modifiable=slippage_modifiable,
            is_blacklisted=is_blacklisted,
            sell_tax=sell_tax,
            buy_tax=buy_tax,
            is_contract_renounced=is_contract_renounced,
            is_potentially_scam=is_potentially_scam,
            updated_at=dt_updated,
            is_suspicious=suspicious_flag
        )

    def get_recent_transactions(self, n: int) -> pd.DataFrame:
        """
        Calls the get_recent_transactions_hours stored procedure with the given parameter
        and returns the result as a pandas DataFrame.

        Args:
            n (int): The number of hours to look back for transactions.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the query.
        """
        try:
            logger.info(f"Fetching recent transactions for the past {n} hours.")

            # Define the SQL call for the stored procedure
            query = "SELECT * FROM get_recent_transactions_hours(%s);"

            # Execute the query and fetch results
            with self.db.get_cursor() as cursor:
                cursor.execute(query, (n,))
                results = cursor.fetchall()

            # Convert the results into a DataFrame
            df = pd.DataFrame(results, columns=[
                "dca", "user_address", "open_input_mint", "open_output_mint",
                "close_input_mint", "close_output_mint", "open_signature", "close_signature",
                "in_amount", "cycle_frequency", "in_amount_per_cycle", "out_amount",
                "open_block_time", "close_block_time", "open_block_number", "close_block_number",
                "open_parent_slot", "close_parent_slot", "open_blockhash", "close_blockhash",
                "open_previous_blockhash", "close_previous_blockhash", "opened_at", "closed_at"
            ])

            logger.info(f"Fetched {len(df)} rows of transactions.")
            return df

        except Exception as e:
            logger.error(f"Error fetching recent transactions: {str(e)}")
            logger.error("Error traceback: ", exc_info=True)
            raise

    def get_token_details(self, solana_address: str) -> Optional[Dict[str, Any]]:
        """
        Fetches token name, decimals, and is_suspicious status from the token_info table
        for the given Solana address.

        Args:
            solana_address (str): The Solana address to query.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with token information, or None if no match is found.
        """
        try:
            logger.info(f"Fetching token info for Solana address: {solana_address}")

            # Define the query
            query = """
                SELECT name, decimals, is_suspicious
                FROM token_info
                WHERE token = %s;
            """

            # Execute the query
            with self.db.get_cursor() as cursor:
                cursor.execute(query, (solana_address,))
                result = cursor.fetchone()

            # If no results, return None
            if not result:
                logger.warning(f"No token info found for Solana address: {solana_address}")
                return None

            # Format the result into a dictionary
            token_info = {
                "name": result["name"],
                "decimals": result["decimals"],
                "is_suspicious": result["is_suspicious"],
            }

            logger.info(f"Token info fetched successfully for {solana_address}: {token_info}")
            return token_info

        except Exception as e:
            logger.error(f"Error fetching token info for {solana_address}: {str(e)}")
            logger.error("Error traceback: ", exc_info=True)
            raise



    def update_token_info_with_rate_limit(
            self,
            api_url: str,
            rate_limit: int,
            api_key: str = ""
    ):
        """Updates token information from DexTools API with rate limiting."""
        try:
            logger.info("Starting update_token_info_with_rate_limit")
            logger.debug(f"Parameters - API URL: {api_url}, Rate limit: {rate_limit}")

            # 1. Collect unique tokens
            logger.info("Collecting unique tokens...")
            try:
                unique_tokens = self.get_unique_mints()
                logger.debug(f"Retrieved {len(unique_tokens)} unique tokens")
                logger.debug(f"Sample tokens: {unique_tokens[:5]}...")  # Log first 5 tokens
            except Exception as e:
                logger.error(f"Error collecting unique tokens: {str(e)}")
                raise

            if not unique_tokens:
                logger.info("No new tokens to process.")
                return

            # 2. Process in batches
            batch_size = min(50, rate_limit)
            logger.info(f"Processing {len(unique_tokens)} tokens in batches of {batch_size}")

            for i in range(0, len(unique_tokens), batch_size):
                try:
                    batch = unique_tokens[i : i + batch_size]
                    logger.debug(f"Processing batch {i//batch_size + 1}, size: {len(batch)}")
                    logger.debug(f"First token in batch: {batch[0]}")

                    if i > 0 and i % rate_limit == 0:
                        logger.info("Rate limit reached. Sleeping for 1 second...")
                        time.sleep(1)

                    valid_tokens = []
                    suspicious_tokens = []

                    for token in batch:
                        try:
                            logger.debug(f"Processing token: {token}")

                            # DexTools API calls
                            base_url = f"{api_url}/v2/token/solana/{token}"
                            headers = {"X-API-KEY": api_key}

                            logger.debug(f"Making base info request to: {base_url}")
                            info_resp = requests.get(base_url, headers=headers)
                            logger.debug(f"Base info response status: {info_resp.status_code}")

                            audit_url = f"{api_url}/v2/token/solana/{token}/audit"
                            logger.debug(f"Making audit request to: {audit_url}")
                            audit_resp = requests.get(audit_url, headers=headers)
                            logger.debug(f"Audit response status: {audit_resp.status_code}")

                            if info_resp.status_code == 200 and audit_resp.status_code == 200:
                                info_data = info_resp.json()
                                audit_data = audit_resp.json()

                                logger.debug("Processing DexTools response...")
                                dex_info = self.process_dextools_response(info_data, audit_data)

                                if dex_info and dex_info.token:
                                    if dex_info.is_suspicious:
                                        suspicious_tokens.append(dex_info)
                                    else:
                                        valid_tokens.append(dex_info)
                                else:
                                    logger.warning(f"No valid dex_info for token {token}")
                                    suspicious_tokens.append(
                                        DexToolsTokenInfo(token=token, is_suspicious=True)
                                    )
                            else:
                                logger.warning(f"DexTools API call failed for {token}")
                                logger.warning(f"Info response: {info_resp.status_code}, Audit response: {audit_resp.status_code}")
                                suspicious_tokens.append(DexToolsTokenInfo(token=token, is_suspicious=True))

                        except Exception as e:
                            logger.error(f"Error processing individual token {token}: {str(e)}")
                            suspicious_tokens.append(DexToolsTokenInfo(token=token, is_suspicious=True))

                        time.sleep(1)

                    # Store results
                    if valid_tokens:
                        try:
                            logger.debug(f"Storing {len(valid_tokens)} valid tokens")
                            self.store_token_info(valid_tokens)
                            logger.info(f"Stored {len(valid_tokens)} valid tokens (batch {i//batch_size + 1})")
                        except Exception as e:
                            logger.error(f"Error storing valid tokens: {str(e)}")

                    if suspicious_tokens:
                        try:
                            logger.debug(f"Storing {len(suspicious_tokens)} suspicious tokens")
                            self.store_token_info(suspicious_tokens)
                            logger.info(f"Stored {len(suspicious_tokens)} suspicious tokens (batch {i//batch_size + 1})")
                        except Exception as e:
                            logger.error(f"Error storing suspicious tokens: {str(e)}")

                except Exception as e:
                    logger.error(f"Error processing batch starting at index {i}: {str(e)}")
                    logger.error("Error traceback: ", exc_info=True)
                    continue  # Continue with next batch even if current fails

                time.sleep(1)

            logger.info("Finished processing all tokens via DexTools.")

        except Exception as e:
            logger.error(f"Fatal error in update_token_info_with_rate_limit: {str(e)}")
            logger.error("Error traceback: ", exc_info=True)
            raise
    def update_incomplete_token_info(self, api_url: str, rate_limit: int, api_key: str = ""):
        """Update token records where symbol or name is null or empty."""
        with self.db.get_cursor() as cur:
            cur.execute("""
                SELECT token 
                FROM token_info 
                WHERE symbol IS NULL OR symbol = '' OR name IS NULL OR name = ''
            """)
            incomplete_tokens = [row['token'] for row in cur.fetchall()]

        if not incomplete_tokens:
            logger.info("No incomplete token records found.")
            return

        logger.info(f"Found {len(incomplete_tokens)} incomplete token records")

        batch_size = min(50, rate_limit)
        for i in range(0, len(incomplete_tokens), batch_size):
            batch = incomplete_tokens[i:i + batch_size]

            if i > 0 and i % rate_limit == 0:
                logger.info("Rate limit reached. Sleeping for 1 second...")
                time.sleep(1)

            valid_tokens = []
            suspicious_tokens = []

            for token in batch:
                try:
                    logger.debug(f"Processing token: {token}")
                    base_url = f"{api_url}/v2/token/solana/{token}"
                    audit_url = f"{base_url}/audit"
                    headers = {"X-API-KEY": api_key}

                    info_resp = requests.get(base_url, headers=headers)
                    audit_resp = requests.get(audit_url, headers=headers)

                    if info_resp.status_code == 200:
                        info_data = info_resp.json()
                    else:
                        logger.warning(f"Info API call failed for token {token}")
                        info_data = {}

                    if audit_resp.status_code == 200:
                        audit_data = audit_resp.json()
                    else:
                        logger.warning(f"Audit API call failed for token {token}")
                        audit_data = {}

                    dex_info = self.process_dextools_response(info_data, audit_data)
                    if dex_info:
                        if dex_info.is_suspicious:
                            suspicious_tokens.append(dex_info)
                        else:
                            valid_tokens.append(dex_info)

                except Exception as e:
                    logger.error(f"Error processing token {token}: {str(e)}", exc_info=True)

                time.sleep(1)

            if valid_tokens:
                self.store_token_info(valid_tokens)
                logger.info(f"Updated {len(valid_tokens)} valid tokens (batch {i // batch_size + 1})")

            if suspicious_tokens:
                self.store_token_info(suspicious_tokens)
                logger.info(f"Updated {len(suspicious_tokens)} suspicious tokens (batch {i // batch_size + 1})")

        logger.info("Finished updating incomplete token records.")

    def call_stored_procedures(self):
        """Call stored procedures to process DCA trends"""
        try:
            with self.db.get_cursor() as cur:
                logger.info("Calling stored procedure: process_dca_transactions_open()")
                cur.execute("CALL process_dca_transactions_open();")
                logger.info("Calling stored procedure: process_dca_transactions_close()")
                cur.execute("CALL process_dca_transactions_close();")
                logger.info("Calling stored procedure: process_dca_transactions_closedca()")
                cur.execute("CALL process_dca_transactions_closedca();")
            logger.info("Stored procedures executed successfully.")
        except Exception as e:
            logger.error(f"Error calling stored procedures: {str(e)}")
            raise Exception(f"Error calling stored procedures: {str(e)}")

    def load_solana_dca_data(self):
        """
        Fetch rows from solana_dca_endandclose_trends and return them as a pandas DataFrame.
        We'll use the context manager from DatabaseConnection to get a cursor.
        """
        query = "SELECT * FROM solana_dca_endandclose;"

        # Use the context manager to execute the query
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()  # List of dicts (RealDictCursor)

        # Convert list of dicts to a DataFrame
        df = pd.DataFrame(rows)
        return df

    def get_most_active_users(self, limit=10):
        """
        Returns a DataFrame of the top 'limit' user addresses by transaction count 
        and total out_amount.
        """
        query = """
            SELECT 
                user_address, 
                COUNT(*) AS tx_count, 
                SUM(out_amount) AS total_out 
            FROM jupiter_dca_transactions
            GROUP BY user_address
            ORDER BY tx_count DESC
            LIMIT %s;
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        return df

    # queries.py

    def get_tokens_most_out_per_day(self, start_date=None, end_date=None):
        base_query = """
            SELECT * FROM jupiter_dca_transactions;
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(base_query)
            rows = cursor.fetchall()

        # If using RealDictCursor, 'rows' is list-of-dicts:
        df = pd.DataFrame(rows)
        # => df should have columns: ["day", "token_name", "tx_count", "total_out"]

        return df


    def get_tokens_out_more_than_in(self, start_date, end_date, top_n=10):
        """
        Analyzes token output amounts within a date range, excluding stablecoins.
        Returns daily volumes for top N tokens by total volume.
        """
        query = """
            SELECT * FROM jupiter_dca_transactions;
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(query, (start_date, end_date, top_n))
            rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=['date', 'token_name', 'volume'])

    def get_top10_tokens_by_outflow(self):
        """
        Returns the top 10 tokens by total sum of out_amount.
        """
        query = """
            SELECT * FROM jupiter_dca_transactions;
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        return df

    def get_daily_transaction_volume(self):
        """
        Returns the total daily volume (sum of out_amount) for all tokens.
        """
        query = """
            SELECT * FROM jupiter_dca_transactions;
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        return df

    def get_active_users_by_date_range(
            self,
            from_date: datetime,
            to_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get most active users within a specific date range based on DCA transactions

        Args:
            from_date (datetime): Start date of the range (inclusive)
            to_date (datetime): End date of the range (inclusive)

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing user_address and transaction_count
        """
        try:
            logger.info(f"Fetching active users from {from_date} to {to_date}")

            with self.db.get_cursor() as cur:
                query = """
                    WITH transactions_in_range AS (
                        SELECT 
                            user_address,
                            open_block_time as block_time
                        FROM jupiter_dca_transactions
                        WHERE open_block_time >= EXTRACT(EPOCH FROM %s)
                        AND open_block_time <= EXTRACT(EPOCH FROM %s)
                        
                        UNION ALL
                        
                        SELECT 
                            user_address,
                            close_block_time as block_time
                        FROM jupiter_dca_transactions
                        WHERE close_block_time >= EXTRACT(EPOCH FROM %s)
                        AND close_block_time <= EXTRACT(EPOCH FROM %s)
                        AND close_block_time IS NOT NULL
                    )
                    SELECT 
                        user_address,
                        COUNT(*) as transaction_count
                    FROM transactions_in_range
                    GROUP BY user_address
                    ORDER BY transaction_count DESC;
                """

                # Pass the dates twice because they're used in both parts of the UNION
                cur.execute(query, (from_date, to_date, from_date, to_date))
                results = cur.fetchall()

                logger.info(f"Found {len(results)} active users between {from_date} and {to_date}")
                logger.debug(f"Top 5 most active users: {results[:5]}")

                return results

        except Exception as e:
            logger.error(f"Error fetching active users: {str(e)}")
            logger.error("Error traceback: ", exc_info=True)
            raise

    def process_dextools_details(self, chain: str, address: str, api_url: str, api_key: str, rate_limit: float = 1.0) -> Optional[DexToolsTokenDetails]:
        """
        Fetches and processes additional token details from DexTools API for a given chain and address.
        """
        try:
            # URLs for the endpoints
            info_url = f"{api_url}/v2/token/{chain}/{address}/info"
            price_url = f"{api_url}/v2/token/{chain}/{address}/price"

            headers = {"X-API-KEY": api_key}

            # Make API requests to fetch info and price
            logger.info(f"Fetching token details for {address} on chain {chain}...")
            logger.debug(f"Requesting info from: {info_url}")
            info_response = requests.get(info_url, headers=headers)

            # Respect rate limits
            time.sleep(rate_limit)

            logger.debug(f"Requesting price from: {price_url}")
            price_response = requests.get(price_url, headers=headers)

            # Handle responses
            if info_response.status_code != 200 or price_response.status_code != 200:
                logger.warning(f"Failed to fetch token details for {address}. Info status: {info_response.status_code}, Price status: {price_response.status_code}")
                return None

            # Parse JSON responses
            info_data = info_response.json()
            price_data = price_response.json()

            # Check for empty data
            if not info_data or not price_data:
                logger.warning(f"Received empty data for token {address}. Info: {info_data}, Price: {price_data}")
                return None

            # Create and return a DexToolsTokenDetails instance
            return DexToolsTokenDetails.from_dict(address, chain, info_data, price_data)

        except Exception as e:
            logger.error(f"Error processing token details for {address}: {str(e)}")
            logger.error("Error traceback: ", exc_info=True)
            return None

# -------------------------------------------------------------------
    def fetch_historical_price(self, token_mint, api_key):
        """
        Fetch the historical price data for a given token using the Birdeye API endpoint
        /defi/history_price. The API call requests the price history from:
            time_from = now - 188700 (seconds)
            time_to   = now

        Returns a DataFrame with historical snapshots sorted by unixTime or None if not found.
        """
        now = int(time.time())
        time_from = now - 188700  # 188700 seconds before now
        url = f"https://public-api.birdeye.so/defi/history_price?address={token_mint}&address_type=token&type=5m&time_from={time_from}&time_to={now}"
        logger.info(f"fetch_historical_price: Requesting history for token '{token_mint}' from {time_from} to {now}.")
        headers = {
            "accept": "application/json",
            "x-chain": "solana",
            "X-API-KEY": api_key
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data.get("success") is True:
                items = data.get("data", {}).get("items", [])
                if items:
                    df = pd.DataFrame(items)
                    df = df.sort_values("unixTime")
                    logger.info(f"fetch_historical_price: Retrieved {len(df)} price snapshots for token '{token_mint}'.")
                    return df
                else:
                    logger.error(f"fetch_historical_price: No items found in API response for token '{token_mint}'. Response: {data}")
                    return None
            else:
                logger.error(f"fetch_historical_price: API returned error for token '{token_mint}'. Response: {data}")
                return None
        except requests.RequestException as e:
            logger.error(f"fetch_historical_price: Request error for token '{token_mint}': {e}")
            return None

    # -------------------------------------------------------------------
    def fetch_token_details(self, chain_id, token_addresses, api_key):
        """
        Fetch token details from the Birdeye token overview endpoint.
        For each token address, it sends a GET request to:
          https://public-api.birdeye.so/defi/token_overview?address={token}
        The chain is specified in the request headers.

        Returns a DataFrame containing the details for all tokens.
        """
        logger.info(f"fetch_token_details: Starting to fetch details for {len(token_addresses)} tokens on chain '{chain_id}' using Birdeye API.")
        token_details_list = []
        # Filter out any None values
        token_addresses = [address for address in token_addresses if address is not None]
        for token in tqdm(token_addresses, desc='Fetching token details', unit='token'):
            url = f"https://public-api.birdeye.so/defi/token_overview?address={token}"
            headers = {
                "accept": "application/json",
                "x-chain": chain_id,
                "X-API-KEY": api_key
            }
            logger.info(f"fetch_token_details: Fetching details for token: {token}")
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                if data.get("success") is True:
                    token_data = data.get("data", {})
                    # Ensure the 'address' field exists for merging later.
                    if "address" not in token_data:
                        token_data["address"] = token
                    token_details_list.append(token_data)
                    logger.info(f"fetch_token_details: Retrieved details for token: {token}")
                else:
                    logger.error(f"fetch_token_details: API returned error for token {token}. Response: {data}")
                #sleep(0.2)  # avoid rate limiting
            except requests.RequestException as e:
                logger.error(f"fetch_token_details: Request error for token {token}: {e}")
                #sleep(0.2)
        token_details_df = pd.DataFrame(token_details_list)
        logger.info("fetch_token_details: Finished fetching all token details.")
        return token_details_df

    def fetch_and_store_token_security(self, tokens: list, api_key: str):
        """
        Calls the Birdeye API in batches for a list of tokens, inserts them in batches of 100,
        and marks tokens as 'undefined' if no security data is found to avoid future API calls.
        """
        url_template = "https://public-api.birdeye.so/defi/token_security?address={}"
        headers = {
            "accept": "application/json",
            "x-chain": "solana",
            "X-API-KEY": api_key
        }

        batch_size = 100
        token_batches = [tokens[i:i + batch_size] for i in range(0, len(tokens), batch_size)]

        for batch_num, batch in enumerate(token_batches, start=1):
            records = []
            undefined_records = []
            logger.info(f"Starting batch {batch_num}: Processing {len(batch)} tokens")

            for token in batch:
                try:
                    url = url_template.format(token)
                    logger.debug(f"Requesting security data for token: {token}")

                    response = requests.get(url, headers=headers)

                    if response.status_code == 555:
                        logger.warning(f"API returned 555 for token {token}. Marking as undefined.")
                        undefined_records.append({"token": token, "is_undefined": True})
                        continue  # Skip to next token

                    response.raise_for_status()
                    data = response.json()

                    if not data.get("success"):
                        logger.warning(f"API call failed for token {token}: {data}")
                        undefined_records.append({"token": token, "is_undefined": True})
                        continue  # Skip this token

                    token_data = data.get("data", {})
                    logger.debug(f"API response for {token}: {json.dumps(token_data, indent=2)}")

                    fields = {
                        "token": token,
                        "creator_address": token_data.get("creatorAddress"),
                        "creator_owner_address": token_data.get("creatorOwnerAddress"),
                        "owner_address": token_data.get("ownerAddress"),
                        "owner_of_owner_address": token_data.get("ownerOfOwnerAddress"),
                        "creation_tx": token_data.get("creationTx"),
                        "creation_time": token_data.get("creationTime"),
                        "creation_slot": token_data.get("creationSlot"),
                        "mint_tx": token_data.get("mintTx"),
                        "mint_time": token_data.get("mintTime"),
                        "mint_slot": token_data.get("mintSlot"),
                        "creator_balance": token_data.get("creatorBalance"),
                        "owner_balance": token_data.get("ownerBalance"),
                        "owner_percentage": token_data.get("ownerPercentage"),
                        "creator_percentage": token_data.get("creatorPercentage"),
                        "metaplex_update_authority": token_data.get("metaplexUpdateAuthority"),
                        "metaplex_owner_update_authority": token_data.get("metaplexOwnerUpdateAuthority"),
                        "metaplex_update_authority_balance": token_data.get("metaplexUpdateAuthorityBalance"),
                        "metaplex_update_authority_percent": token_data.get("metaplexUpdateAuthorityPercent"),
                        "mutable_metadata": token_data.get("mutableMetadata"),
                        "top10_holder_balance": token_data.get("top10HolderBalance"),
                        "top10_holder_percent": token_data.get("top10HolderPercent"),
                        "top10_user_balance": token_data.get("top10UserBalance"),
                        "top10_user_percent": token_data.get("top10UserPercent"),
                        "is_true_token": token_data.get("isTrueToken"),
                        "fake_token": token_data.get("fakeToken"),
                        "total_supply": token_data.get("totalSupply"),
                        "freezeable": token_data.get("freezeable"),
                        "freeze_authority": token_data.get("freezeAuthority"),
                        "transfer_fee_enable": token_data.get("transferFeeEnable"),
                        "transfer_fee_data": token_data.get("transferFeeData"),
                        "is_token_2022": token_data.get("isToken2022"),
                        "non_transferable": token_data.get("nonTransferable"),
                        "jup_strict_list": token_data.get("jupStrictList"),
                        "pre_market_holder": json.dumps(token_data.get("preMarketHolder", None)),
                        "lock_info": json.dumps(token_data.get("lockInfo", None)),
                        "is_undefined": False  # Defined token
                    }

                    records.append(fields)

                except requests.exceptions.RequestException as e:
                    logger.error(f"API request error for token {token}: {str(e)}")
                    undefined_records.append({"token": token, "is_undefined": True})
                    continue  # Skip this token

                except Exception as e:
                    logger.error(f"Unexpected error for token {token}: {str(e)}", exc_info=True)
                    undefined_records.append({"token": token, "is_undefined": True})
                    continue  # Skip this token

            # Insert valid records
            if records:
                try:
                    with self.db.get_cursor() as cursor:
                        query = """
                            INSERT INTO token_security (
                                token, creator_address, creator_owner_address, owner_address, owner_of_owner_address,
                                creation_tx, creation_time, creation_slot, mint_tx, mint_time, mint_slot,
                                creator_balance, owner_balance, owner_percentage, creator_percentage,
                                metaplex_update_authority, metaplex_owner_update_authority,
                                metaplex_update_authority_balance, metaplex_update_authority_percent,
                                mutable_metadata, top10_holder_balance, top10_holder_percent,
                                top10_user_balance, top10_user_percent, is_true_token, fake_token,
                                total_supply, pre_market_holder, lock_info, freezeable, freeze_authority,
                                transfer_fee_enable, transfer_fee_data, is_token_2022, non_transferable,
                                jup_strict_list, is_undefined, updated_at
                            )
                            VALUES (
                                %(token)s, %(creator_address)s, %(creator_owner_address)s, %(owner_address)s, %(owner_of_owner_address)s,
                                %(creation_tx)s, %(creation_time)s, %(creation_slot)s, %(mint_tx)s, %(mint_time)s, %(mint_slot)s,
                                %(creator_balance)s, %(owner_balance)s, %(owner_percentage)s, %(creator_percentage)s,
                                %(metaplex_update_authority)s, %(metaplex_owner_update_authority)s,
                                %(metaplex_update_authority_balance)s, %(metaplex_update_authority_percent)s,
                                %(mutable_metadata)s, %(top10_holder_balance)s, %(top10_holder_percent)s,
                                %(top10_user_balance)s, %(top10_user_percent)s, %(is_true_token)s, %(fake_token)s,
                                %(total_supply)s, %(pre_market_holder)s, %(lock_info)s, %(freezeable)s, %(freeze_authority)s,
                                %(transfer_fee_enable)s, %(transfer_fee_data)s, %(is_token_2022)s, %(non_transferable)s,
                                %(jup_strict_list)s, %(is_undefined)s, NOW()
                            )
                            ON CONFLICT (token) DO NOTHING;
                        """
                        cursor.executemany(query, records)  # Batch insert

                    logger.info(f"Batch {batch_num} successfully inserted into the database.")

                except Exception as e:
                    logger.error(f"Database insert error: {str(e)}", exc_info=True)

            # Insert undefined tokens
            if undefined_records:
                try:
                    with self.db.get_cursor() as cursor:
                        query = """
                            INSERT INTO token_security (token, is_undefined, updated_at)
                            VALUES (%(token)s, %(is_undefined)s, NOW())
                            ON CONFLICT (token) DO NOTHING;
                        """
                        cursor.executemany(query, undefined_records)  # Batch insert

                    logger.info(f"Marked {len(undefined_records)} tokens as 'undefined'.")

                except Exception as e:
                    logger.error(f"Database insert error for undefined tokens: {str(e)}", exc_info=True)

            logger.info("Sleeping for 1 second to prevent DB overload...")
            time.sleep(5)
