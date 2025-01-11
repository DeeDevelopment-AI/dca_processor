import json
import logging
import requests
import traceback
from tkinter import filedialog
import tkinter as tk

from typing import Optional, Dict, List, Any

from config import load_config
from processors.block_processor import BlockProcessor, parse_block_data, parse_block_data_file  # Updated to use new code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_and_process_block(block_number: int) -> Optional[List[Dict[str, Any]]]:
    """
    1. Load config and instantiate a new BlockProcessor.
    2. Test some alternate method names (for debugging).
    3. Fetch the block from Quicknode.
    4. Parses the block (internally in BlockProcessor).
    5. Returns the list of matched DCA transactions.
    """
    try:
        config = load_config()
        block_processor = BlockProcessor(config.quicknode)

        # Actually fetch the block
        logger.info("\nFetching block from Quicknode.")
        block_data = block_processor.fetch_block(block_number)
        if not block_data:
            logger.error("Failed to fetch block or parse it.")
            return None

        logger.info("Successfully fetched block from Quicknode.")
        return inspect_block_data_api(block_data, block_processor)

    except Exception as e:
        logger.error(f"Error fetching block: {e}")
        logger.exception(e)
        return None


def inspect_block_data_api(block_data: Dict[str, Any],
                           block_processor: Optional[BlockProcessor] = None) -> List[Dict[str, Any]]:
    """
    Logs high-level block info and returns the list of
    matched DCA transactions from block_data["matchedTransactions"].

    :param block_data: A dictionary with at least:
      {
        "blockHeight": ...,
        "blockTime": ...,
        "blockhash": ...,
        "parentSlot": ...,
        "previousBlockhash": ...,
        "matchedTransactions": [...matched DCA transactions...]
      }
    :param block_processor: An optional BlockProcessor to get a typed BlockInfo object.
    :return: A list of DCA transaction dictionaries, or empty if none found.
    """
    try:
        # Validate block_data structure
        if "matchedTransactions" not in block_data or not isinstance(block_data["matchedTransactions"], list):
            logger.warning("No 'matchedTransactions' field found in block_data.")
            return []

        # If block_processor is not provided, initialize one using the config
        if not block_processor:
            config = load_config()
            block_processor = BlockProcessor(config.quicknode)

        # Convert block_data to a BlockInfo object for detailed logging
        block_info = block_processor.get_block_info(block_data)

        # logger.info("\n=== Processed Block Info ===")
        # logger.info(f"  Height: {block_info.block_height}")
        # logger.info(f"  Time: {block_info.block_time}")
        # logger.info(f"  Blockhash: {block_info.blockhash}")
        # logger.info(f"  Parent slot: {block_info.parent_slot}")
        # logger.info(f"  Previous blockhash: {block_info.previous_blockhash}")

        # Extract and log matched transactions
        dca_transactions = block_data["matchedTransactions"]
        logger.info(f"\nDCA Transactions found: {len(dca_transactions)}")

        # Optionally log each transaction's key fields
        for idx, tx in enumerate(dca_transactions, start=1):
            logger.info(f"\nDCA Transaction #{idx} ===============")
            logger.info(f"  Signature:       {tx.get('signature')}")
            logger.info(f"  User:            {tx.get('user')}")
            logger.info(f"  Input Mint:      {tx.get('inputMint')}")
            logger.info(f"  Output Mint:     {tx.get('outputMint')}")
            logger.info(f"  In Amount:       {tx.get('inAmount')}")
            logger.info(f"  Amt Per Cycle:   {tx.get('inAmountPerCycle')}")
            logger.info(f"  Cycle Frequency: {tx.get('cycleFrequency')}")

        return dca_transactions

    except Exception as e:
        logger.error(f"Error inspecting block data: {e}")
        logger.exception(e)
        return []



def main():


    # 2) Fetch the same block from API
    logger.info("\n=== Testing block fetching and parsing from API ===")
    block_number = 303787383  # Example block
    api_results = fetch_and_process_block(block_number)



if __name__ == "__main__":
    main()

#%%
