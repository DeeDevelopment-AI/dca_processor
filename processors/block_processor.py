import json
import logging
from typing import Optional, Dict, Any, List
import traceback
import base58  # pip install base58

# Keep your original config, models, etc.
from config import QuicknodeConfig
from models import BlockInfo

# NEW: We'll use solana-py’s Client for fetching blocks
from solana.rpc.api import Client

logger = logging.getLogger(__name__)

# Constants from your JavaScript code
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
DCA_PROGRAM_ID_PREFIX = "DCA265"
OPEN_DCA_LOG = "Program log: Instruction: OpenDca"

###############################################################################
# Utility Functions
###############################################################################

def decode_base58(encoded: str) -> bytes:
    """
    Decodes a base58 string into bytes. Similar to your JavaScript `decodeBase58`.
    Here, we use the Python `base58` package for convenience.
    """
    return base58.b58decode(encoded)

def parse_little_endian(hex_string: str) -> int:
    """
    Reverse the byte order for little-endian interpretation, then convert to int.
    """
    # Strip out any non-hex characters (just in case)
    clean_hex = "".join([c for c in hex_string if c in "0123456789abcdefABCDEF"])
    # Split into bytes and reverse
    bytes_list = [clean_hex[i : i + 2] for i in range(0, len(clean_hex), 2)]
    bytes_list.reverse()
    reversed_hex = "".join(bytes_list)
    return int(reversed_hex, 16)

###############################################################################
# Parsing Logic
###############################################################################

import json
from typing import List, Dict, Any, Optional


def parse_block_data_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses a block file to extract matching DCA transactions.

    :param file_path: Path to the JSON file containing block data.
    :return: A list of matching DCA transactions.
    """
    try:
        # Load the block data from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            block_data = json.load(f)

        if not isinstance(block_data, list) or len(block_data) == 0:
            raise ValueError("Invalid block data structure. Expected a list of blocks.")

        # Process each block in the file
        all_matched_transactions = []

        for block in block_data:
            # Ensure the block contains necessary fields
            if not all(key in block for key in ["blockHeight", "blockTime", "blockhash", "parentSlot", "transactions"]):
                continue

            # Construct block info
            block_info = {
                "blockHeight": block["blockHeight"],
                "blockTime": block["blockTime"],
                "blockhash": block["blockhash"],
                "parentSlot": block["parentSlot"],
                "previousBlockhash": block.get("previousBlockhash", None),
            }

            # Parse transactions
            matched_tx = []
            for tx_with_meta in block.get("transactions", []):
                if is_dca_transaction(tx_with_meta):
                    parsed_list = parse_dca_transaction(tx_with_meta, block_info)
                    matched_tx.extend(parsed_list)

            # Add matching transactions from the current block
            if matched_tx:
                all_matched_transactions.extend(matched_tx)

        return all_matched_transactions

    except Exception as e:
        print(f"Error processing block file: {e}")
        import traceback
        traceback.print_exc()
        return []

def is_dca_transaction(tx) -> bool:
    """
    Checks for:
      - A log containing 'Program log: Instruction: OpenDca'
      - An instruction whose programId contains 'DCA265'
      - That instruction has at least 12 accounts (in the compiled sense)
    """
    logs = tx.meta.log_messages if tx.meta and tx.meta.log_messages else []
    # 1) Check if logs contain the "OpenDca" message
    if not any(OPEN_DCA_LOG in log for log in logs):
        return False

    # 2) Check instructions
    if not hasattr(tx.transaction, 'message'):
        return False

    compiled_instructions = tx.transaction.message.instructions
    # `compiled_instructions` will be a list of `UiCompiledInstruction` objects
    for ix in compiled_instructions:
        # Each ix has `program_id_index`, `accounts`, `data` (all base58).
        program_id_index = ix.program_id_index
        # The actual program ID is found in the message.account_keys array
        program_pubkey = tx.transaction.message.account_keys[program_id_index]

        # Convert the program_pubkey to a string and see if it contains the prefix
        if DCA_PROGRAM_ID_PREFIX in str(program_pubkey):
            # If the number of accounts is at least 12, we say it's a DCA transaction
            if len(ix.accounts) > 11:
                return True

    return False


def parse_dca_transaction(tx, block_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    parsed_results = []
    try:
        #signature = tx.transaction.signatures[0] if tx.transaction.signatures else None
        # Assuming tx.transaction.signatures is the list you are processing:
        signatures = tx.transaction.signatures if tx.transaction.signatures else []

        # Convert the Signature objects into a list of strings
        signature_list = [str(sig) for sig in signatures]

        # Clean up any unnecessary formatting (if needed)
        signature_array = [sig.strip("Signature(").strip(",)") for sig in signature_list]
        #print("Transaction Signature:", signature)

        # Calculate blockNumber = parentSlot + 1
        block_number = block_info["parentSlot"] + 1

        # print("Block Information:", {
        #     "blockHeight": block_info["blockHeight"],
        #     "blockTime": block_info["blockTime"],
        #     "parentSlot": block_info["parentSlot"],
        #     "blockNumber": block_number
        # })

        instructions = tx.transaction.message.instructions  # UiCompiledInstruction list

        for ix in instructions:
            program_id_index = ix.program_id_index
            program_pubkey = tx.transaction.message.account_keys[program_id_index]

            # Check if this instruction is from the DCA program
            if DCA_PROGRAM_ID_PREFIX not in str(program_pubkey):
                continue

            # We expect either 12 or 13 accounts in the compiled sense
            if len(ix.accounts) not in (12, 13):
                continue

            # Now we map the compiled `ix.accounts` indices into actual PubKeys
            account_pubkeys = [
                str(tx.transaction.message.account_keys[a_idx])
                for a_idx in ix.accounts
            ]

            # Mirror the JS logic for account layout
            if len(ix.accounts) == 13:
                dca = account_pubkeys[0]
                user = account_pubkeys[2]
                input_mint = account_pubkeys[3]
                output_mint = account_pubkeys[4]
            else:  # 12
                dca = account_pubkeys[0]
                user = account_pubkeys[1]
                input_mint = account_pubkeys[2]
                output_mint = account_pubkeys[3]

            # print("User Account:", user)
            # print("Input Mint:", input_mint)
            # print("Output Mint:", output_mint)

            # Retrieve decimals from postTokenBalances if available
            decimals = 0
            if tx.meta and tx.meta.post_token_balances:
                post_balances = tx.meta.post_token_balances
                out_token_balance = next(
                    (
                        b
                        for b in post_balances
                        # b.mint and b.owner are strings (PublicKey)
                        if b.mint == str(input_mint) and b.owner == str(user)
                    ),
                    None
                )
                if out_token_balance and out_token_balance.ui_token_amount:
                    decimals = out_token_balance.ui_token_amount.decimals

            # If inputMint is the native SOL mint, set decimals=9
            if input_mint == "So11111111111111111111111111111111111111112":
                decimals = 9

            # Decode the base58 instruction data
            data = ix.data  # base58 string
            bytedata = decode_base58(data)
            #print("Decoded Data (Base58):", list(bytedata))

            # Convert bytedata to hex string for easy substring operations
            hex_string = bytedata.hex()
            #print("Hex String:", hex_string)

            # Per your offsets:
            in_amount_bytes_hex = hex_string[16 * 2 : 24 * 2]
            in_amount_per_cycle_bytes_hex = hex_string[24 * 2 : 32 * 2]
            cycle_frequency_bytes_hex = hex_string[32 * 2 : 40 * 2]

            # print("inAmountbytes:", in_amount_bytes_hex)
            # print("cycleFrequencyBytes:", cycle_frequency_bytes_hex)
            # print("inAmountPerCycleBytes:", in_amount_per_cycle_bytes_hex)

            if not in_amount_bytes_hex or not in_amount_per_cycle_bytes_hex or not cycle_frequency_bytes_hex:
                raise ValueError("Failed to extract byte ranges.")

            # Convert from little-endian hex to integer
            in_amount = parse_little_endian(in_amount_bytes_hex)
            cycle_frequency = parse_little_endian(cycle_frequency_bytes_hex)
            in_amount_per_cycle = parse_little_endian(in_amount_per_cycle_bytes_hex)

            # Scale amounts
            UIinAmount = str(float(in_amount) / (10**decimals))
            UIinAmountPerCycle = str(float(in_amount_per_cycle) / (10**decimals))

            # print("Cycle Frequency:", cycle_frequency)
            # print("In Amount:", in_amount)
            # print("In Amount Per Cycle:", in_amount_per_cycle)

            parsed_results.append({
                "dca": dca,
                "signature": signature_array,
                "user": user,
                "inputMint": input_mint,
                "outputMint": output_mint,
                "cycleFrequency": str(cycle_frequency),
                "inAmount": UIinAmount,
                "inAmountPerCycle": UIinAmountPerCycle,
                "blockHeight": block_info["blockHeight"],
                "blockTime": block_info["blockTime"],
                "blockhash": str(block_info["blockhash"]),
                "parentSlot": block_info["parentSlot"],
                "previousBlockhash": str(block_info["previousBlockhash"]),
                "blockNumber": block_number
            })

    except Exception as e:
        print("Error parsing DCA transaction:", str(e))
        import traceback
        traceback.print_exc()

    return parsed_results


def parse_block_data(block_data: Any) -> Optional[Dict[str, Any]]:
    """
    Main parsing function, analogous to your JavaScript `main(stream)`.
    Expects block_data in a format similar to the `block_resp.value` returned by
    solana_client.get_block().
    """
    try:
        # If there's no transactions, return None
        if not block_data or not block_data.transactions:
            return None

        # Build our blockInfo object
        block_info = {
            "blockHeight": block_data.block_height,
            "blockTime": block_data.block_time,
            "blockhash": block_data.blockhash,
            "parentSlot": block_data.parent_slot,
            "previousBlockhash": block_data.previous_blockhash,
        }

        # Filter to only the DCA transactions
        matched_tx = []
        for tx_with_meta in block_data.transactions:
            if is_dca_transaction(tx_with_meta):
                # parseDcaTransaction returns a list
                parsed_list = parse_dca_transaction(tx_with_meta, block_info)
                matched_tx.extend(parsed_list)

        if len(matched_tx) == 0:
            return None

        logger.info("\n=== Processed Block Info ===")
        logger.info(f"  Height: {parsed_list}")

        return {"matchedTransactions": matched_tx}

    except Exception as error:
        print("Error in parse_block_data:", str(error))
        return {
            "error": str(error),
            "stack": traceback.format_exc()
        }


class BlockProcessor:
    def __init__(self, config: QuicknodeConfig):
        """
        We keep the same initialization.
        config.api_url => used as the Solana RPC endpoint
        config.api_key => typically for requests-based auth,
                          but solana-py doesn't handle that natively.
        """
        self.config = config
        logger.info(f"Initialized BlockProcessor with endpoint: {config.api_url}")

    def fetch_block(self, block_number: int) -> Optional[Dict]:
        """
        This function fetches the block by slot (using solana-py),
        then calls `parse_block_data` to parse out DCA transactions.
        """
        solana_client = Client("https://yolo-sly-fire.solana-mainnet.quiknode.pro/ea87b760743f2193b31af6054b169e5a2383ad49/")

        block_resp = solana_client.get_block(
            block_number,
            max_supported_transaction_version=0,
        )

        block_data = block_resp.value
        if block_data is None:
            print(f"No block found for slot {block_number}")
            return None

        # Now parse it using the logic from your JS code
        result = parse_block_data(block_data)
        return result

    def get_block_info(self, block_data: Dict) -> BlockInfo:
        """
        Create BlockInfo from block data.
        We do NOT change this signature or return type—used elsewhere.
        """
        return BlockInfo(
            block_height=block_data.get('blockHeight'),
            block_time=block_data.get('blockTime'),
            blockhash=block_data.get('blockhash'),
            parent_slot=block_data.get('parentSlot'),
            previous_blockhash=block_data.get('previousBlockhash')
        )
