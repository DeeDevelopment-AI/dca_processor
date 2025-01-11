import logging
import base58
from typing import Dict, List
from models import BlockInfo

logger = logging.getLogger(__name__)

# Constants
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
DCA_PROGRAM_ID_PREFIX = "DCA265"
OPEN_DCA_LOG = "Program log: Instruction: OpenDca"


def decode_base58(encoded: str) -> bytes:
    """Decode base58 string to bytes"""
    try:
        return base58.b58decode(encoded)
    except Exception as e:
        logger.error(f"Error decoding base58: {e}")
        return bytes()


def parse_little_endian(hex_string: str) -> int:
    """Parse little-endian hex string to integer"""
    clean_hex = ''.join(filter(str.isalnum, hex_string))
    reversed_hex = ''.join(reversed([clean_hex[i:i+2] for i in range(0, len(clean_hex), 2)]))
    return int(reversed_hex, 16)


def is_dca_transaction(tx: Dict) -> bool:
    """Check if transaction is a DCA transaction"""
    try:
        # Check for both OpenDca and OpenDcaV2 in logs
        logs = tx.get('meta', {}).get('logMessages', [])
        dca_logs = ["Program log: Instruction: OpenDca", "Program log: Instruction: OpenDcaV2"]

        # Debug logging
        logger.debug(f"Checking transaction logs: {logs}")

        has_dca_log = any(any(dca_log in log for dca_log in dca_logs) for log in logs)
        if not has_dca_log:
            logger.debug("No DCA logs found in transaction")
            return False

        # Check for DCA program ID in instructions
        instructions = tx.get('transaction', {}).get('message', {}).get('instructions', [])

        # Debug logging
        logger.debug(f"Checking instructions: {instructions}")

        for ix in instructions:
            program_id = ix.get('programId', '')
            accounts_length = len(ix.get('accounts', []))
            logger.debug(f"Instruction program ID: {program_id}, accounts length: {accounts_length}")

            if program_id.startswith(DCA_PROGRAM_ID_PREFIX) and accounts_length > 11:
                logger.debug("Found matching DCA instruction")
                return True

        logger.debug("No matching DCA instructions found")
        return False

    except Exception as e:
        logger.error(f"Error in is_dca_transaction: {e}")
        return False


def parse_dca_transaction(tx: Dict, block_info: BlockInfo) -> List[Dict]:
    """Parse DCA transaction and extract relevant information"""
    parsed_results = []
    try:
        signature = tx['transaction']['signatures'][0]
        block_number = block_info.parent_slot + 1

        for instruction in tx['transaction']['message']['instructions']:
            if not instruction['programId'].startswith(DCA_PROGRAM_ID_PREFIX):
                continue

            accounts = instruction['accounts']
            if len(accounts) not in (12, 13):
                continue

            # Parse account information based on length
            if len(accounts) == 13:
                dca, user, input_mint, output_mint = accounts[0], accounts[2], accounts[3], accounts[4]
            else:
                dca, user, input_mint, output_mint = accounts[0], accounts[1], accounts[2], accounts[3]

            # Get decimals from post token balances
            post_balances = tx.get('meta', {}).get('postTokenBalances', [])
            out_token_balance = next(
                (balance for balance in post_balances
                 if balance['mint'] == input_mint and balance['owner'] == user),
                None
            )

            decimals = out_token_balance.get('uiTokenAmount', {}).get('decimals', 0)
            if input_mint == "So11111111111111111111111111111111111111112":
                decimals = 9

            # Parse instruction data
            data = instruction['data']
            byte_data = decode_base58(data)
            hex_string = byte_data.hex()

            # Extract and parse specific byte ranges
            in_amount_bytes = hex_string[32:48]
            cycle_frequency_bytes = hex_string[64:80]
            in_amount_per_cycle_bytes = hex_string[48:64]

            # Parse values
            cycle_frequency = parse_little_endian(cycle_frequency_bytes)
            in_amount = parse_little_endian(in_amount_bytes)
            in_amount_per_cycle = parse_little_endian(in_amount_per_cycle_bytes)

            # Convert to UI values
            ui_in_amount = in_amount / (10 ** decimals)
            ui_in_amount_per_cycle = in_amount_per_cycle / (10 ** decimals)

            parsed_results.append({
                'dca': dca,
                'signature': signature,
                'user': user,
                'input_mint': input_mint,
                'output_mint': output_mint,
                'cycle_frequency': cycle_frequency,
                'in_amount': ui_in_amount,
                'in_amount_per_cycle': ui_in_amount_per_cycle,
                'block_height': block_info.block_height,
                'block_time': block_info.block_time,
                'blockhash': block_info.blockhash,
                'parent_slot': block_info.parent_slot,
                'previous_blockhash': block_info.previous_blockhash,
                'block_number': block_number
            })

    except Exception as e:
        logger.error(f"Error parsing DCA transaction: {e}")
        logger.exception(e)

    return parsed_results
