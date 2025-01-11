from typing import List, Optional, Dict, Any
from models import BlockGap, DCATransaction, DexToolsTokenInfo
from db.connection import DatabaseConnection
import time
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

    def process_dextools_response(self, info_data: dict, audit_data: dict) -> DexToolsTokenInfo:
        """
        Given two responses:
          - info_data: from GET /v2/token/{chain}/{address}
          - audit_data: from GET /v2/token/{chain}/{address}/audit
        Return a single DexToolsTokenInfo object capturing all fields.
        """
        if "data" not in info_data:
            logger.warning("Base token info missing 'data' key.")
            return None
        base = info_data["data"]

        # If "data" is missing from the audit response, use an empty dict or handle gracefully
        audit = audit_data.get("data", {})

        # Parse base fields
        token_address = strip_control_chars(base.get("address"))
        token_name = strip_control_chars(base.get("name"))
        token_symbol = strip_control_chars(base.get("symbol"))
        token_logo = base.get("logo")
        token_desc = base.get("description")
        creation_time = base.get("creationTime")    # "2024-11-13T20:44:16.067Z"
        creation_block = base.get("creationBlock")
        decimals = base.get("decimals")
        social_info = base.get("socialInfo", {})

        # Convert creation_time string -> datetime if needed
        dt_creation = None
        if creation_time:
            try:
                dt_creation = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
            except Exception:
                logger.warning(f"Failed to parse creationTime: {creation_time}")

        # Parse audit fields
        is_open_source = audit.get("isOpenSource")
        is_honeypot = audit.get("isHoneypot")
        is_mintable = audit.get("isMintable")
        is_proxy = audit.get("isProxy")
        slippage_modifiable = audit.get("slippageModifiable")
        is_blacklisted = audit.get("isBlacklisted")
        sell_tax = audit.get("sellTax", {})  # { "min": null, "max": null, "status": "unknown" }
        buy_tax = audit.get("buyTax", {})
        is_contract_renounced = audit.get("isContractRenounced")
        is_potentially_scam = audit.get("isPotentiallyScam")
        updated_at_str = audit.get("updatedAt")  # "2024-11-13T21:11:41.249Z"

        dt_updated = None
        if updated_at_str:
            try:
                dt_updated = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
            except Exception:
                logger.warning(f"Failed to parse updatedAt: {updated_at_str}")

        # Decide if token is suspicious
        # e.g., if is_potentially_scam == "yes" or is_blacklisted == "yes"
        suspicious_flag = (is_potentially_scam == "yes" or is_blacklisted == "yes")

        dex_info = DexToolsTokenInfo(
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

        return dex_info


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
        """Update token records where symbol is null or empty"""
        with self.db.get_cursor() as cur:
            cur.execute("""
                SELECT token 
                FROM token_info 
                WHERE symbol IS NULL OR symbol = ''
                OR name IS NULL OR name = ''
            """)
            incomplete_tokens = [row['token'] for row in cur.fetchall()]

        if not incomplete_tokens:
            logger.info("No incomplete token records found.")
            return

        logger.info(f"Found {len(incomplete_tokens)} incomplete token records")

        # Process in batches
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
                    base_url = f"{api_url}/v2/token/solana/{token}"
                    audit_url = f"{api_url}/v2/token/solana/{token}/audit"
                    headers = {"X-API-KEY": api_key}

                    info_resp = requests.get(base_url, headers=headers)
                    audit_resp = requests.get(audit_url, headers=headers)

                    if info_resp.status_code == 200 and audit_resp.status_code == 200:
                        info_data = info_resp.json()
                        audit_data = audit_resp.json()

                        dex_info = self.process_dextools_response(info_data, audit_data)
                        if dex_info and dex_info.token:
                            if dex_info.is_suspicious:
                                suspicious_tokens.append(dex_info)
                            else:
                                valid_tokens.append(dex_info)
                        else:
                            suspicious_tokens.append(DexToolsTokenInfo(token=token, is_suspicious=True))
                    else:
                        logger.warning(f"DexTools call failed for {token}")
                        suspicious_tokens.append(DexToolsTokenInfo(token=token, is_suspicious=True))

                except Exception as e:
                    logger.error(f"Error processing token {token}: {str(e)}")
                    suspicious_tokens.append(DexToolsTokenInfo(token=token, is_suspicious=True))

                time.sleep(1)

            # Store the results
            if valid_tokens:
                try:
                    self.store_token_info(valid_tokens)
                    logger.info(f"Updated {len(valid_tokens)} valid tokens (batch {i//batch_size + 1})")
                except Exception as e:
                    logger.error(f"Error storing valid tokens: {str(e)}")

            if suspicious_tokens:
                try:
                    self.store_token_info(suspicious_tokens)
                    logger.info(f"Updated {len(suspicious_tokens)} suspicious tokens (batch {i//batch_size + 1})")
                except Exception as e:
                    logger.error(f"Error storing suspicious tokens: {str(e)}")

            time.sleep(1)

        logger.info("Finished updating incomplete token records.")

    def call_stored_procedures(self):
        """Call stored procedures to process DCA trends"""
        try:
            with self.db.get_cursor() as cur:
                logger.info("Calling stored procedure: process_dca_transactions_open()")
                cur.execute("CALL process_dca_transactions_open();")
                logger.info("Calling stored procedure: process_dca_transactions_close()")
                cur.execute("CALL process_dca_transactions_close();")
            logger.info("Stored procedures executed successfully.")
        except Exception as e:
            logger.error(f"Error calling stored procedures: {str(e)}")
            raise Exception(f"Error calling stored procedures: {str(e)}")

    def load_solana_dca_data(self):
        """
        Fetch rows from solana_dca_endandclose_trends and return them as a pandas DataFrame.
        We'll use the context manager from DatabaseConnection to get a cursor.
        """
        query = "SELECT * FROM solana_dca_endandclose_trends;"

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
            FROM solana_dca_endandclose_trends
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
            SELECT 
                DATE_TRUNC('day', to_timestamp(scet.block_time)) AS day,
                COALESCE(ti.name, ti.symbol, scet.output_mint) AS token_name,
                COUNT(*) AS tx_count,
                SUM(scet.out_amount) AS total_out
            FROM solana_dca_endandclose_trends scet
            INNER JOIN token_info ti 
                ON scet.output_mint = ti.token
            WHERE ti.is_suspicious = false
              AND scet.output_mint NOT IN ('So1111111111', 'MintUSDC', 'MintUSDT')
        """

        # (Apply date filters, etc.)
        base_query += """
            GROUP BY 1,2
            ORDER BY day ASC, tx_count DESC
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
            WITH daily_token_volumes AS (
                SELECT 
                    DATE(block_time) as trade_date,
                    output_mint,
                    SUM(output_amount) as daily_volume
                FROM solana_dca_endandclose_trends
                WHERE block_time BETWEEN %s AND %s
                AND output_mint NOT IN (
                    'So11111111111111111111111111111111111111112',  -- SOL
                    'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',  -- USDT
                    'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'   -- USDC
                )
                GROUP BY DATE(block_time), output_mint
            ),
            token_total_volumes AS (
                SELECT 
                    output_mint,
                    SUM(daily_volume) as total_volume
                FROM daily_token_volumes
                GROUP BY output_mint
                ORDER BY total_volume DESC
                LIMIT %s
            )
            SELECT 
                dtv.trade_date,
                COALESCE(ti.name, ti.symbol, dtv.output_mint) as token_name,
                dtv.daily_volume
            FROM daily_token_volumes dtv
            INNER JOIN token_total_volumes ttv 
                ON dtv.output_mint = ttv.output_mint
            INNER JOIN token_info ti 
                ON dtv.output_mint = ti.token
            WHERE ti.is_suspicious = false
            ORDER BY dtv.trade_date, total_volume DESC;
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
            SELECT 
                COALESCE(ti.name, ti.symbol, scet.output_mint) AS token_name,
                SUM(scet.out_amount) AS total_out
            FROM solana_dca_endandclose_trends scet
            INNER JOIN token_info ti
                   ON scet.output_mint = ti.token
            WHERE ti.is_suspicious = false
            GROUP BY token_name
            ORDER BY total_out DESC
            LIMIT 10;
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
            SELECT
                DATE_TRUNC('day', to_timestamp(block_time)) AS day,
                SUM(out_amount) AS daily_volume
            FROM solana_dca_endandclose_trends
            GROUP BY 1
            ORDER BY day;
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