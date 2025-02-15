# models.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class TokenMetrics:
    address: str
    name: str
    price: float
    mcap: float
    holders: int
    output_count: int
    input_count: int
    distinct_output_users: int
    distinct_input_users: int
    total_out_amount: float
    total_volume: float
    variation_5m: float
    variation_1h: float
    variation_6h: float
    variation_24h: float

@dataclass
class BlockInfo:
    block_height: int
    block_time: int
    blockhash: str
    parent_slot: int
    previous_blockhash: str

@dataclass
class BlockGap:
    start_block: int
    end_block: int

@dataclass
class DCATransaction:
    # Matching the solana_dca_endandclose_trends table
    dca: str
    user_address: str
    input_mint: str
    output_mint: str
    out_amount: float
    block_time: int
    block_number: int
    signature: str
    parent_slot: int
    blockhash: str
    previous_blockhash: str
    created_at: Optional[datetime] = None  # If you want to handle DB default

    @classmethod
    def from_dict(cls, data: Dict) -> 'DCATransaction':
        """
        Create DCATransaction from dictionary keys that match
        the solana_dca_endandclose_trends columns.
        """
        return cls(
            dca=data['dca'],
            user_address=data['user_address'],
            input_mint=data['input_mint'],
            output_mint=data['output_mint'],
            out_amount=float(data['out_amount']),
            block_time=data['block_time'],
            block_number=data['block_number'],
            signature=data['signature'],
            parent_slot=data['parent_slot'],
            blockhash=data['blockhash'],
            previous_blockhash=data['previous_blockhash'],
            # created_at might come back as string, datetime, or null from the DB
            # Adjust parsing accordingly:
            created_at=data.get('created_at')
        )


@dataclass
class DexToolsTokenInfo:
    token: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    logo: Optional[str] = None
    description: Optional[str] = None
    creation_time: Optional[datetime] = None
    creation_block: Optional[int] = None
    decimals: Optional[int] = None

    social_info: Optional[Dict[str, Any]] = field(default_factory=dict)
    is_open_source: Optional[str] = None
    is_honeypot: Optional[str] = None
    is_mintable: Optional[str] = None
    is_proxy: Optional[str] = None
    slippage_modifiable: Optional[str] = None
    is_blacklisted: Optional[str] = None
    sell_tax: Optional[Dict[str, Any]] = field(default_factory=dict)
    buy_tax: Optional[Dict[str, Any]] = field(default_factory=dict)
    is_contract_renounced: Optional[str] = None
    is_potentially_scam: Optional[str] = None
    updated_at: Optional[datetime] = None

    is_suspicious: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> 'DexToolsTokenInfo':
        """
        Create DexToolsTokenInfo from a dictionary that contains
        all relevant fields (like from DexTools /v2/token response).
        Adjust field mappings as needed if your `data` has different keys.
        """
        # We'll parse creation_time and updated_at from ISO strings if present
        def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
            if not dt_str:
                return None
            # DexTools might return "2024-11-13T20:44:16.067Z", so we replace 'Z' with UTC offset
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))

        # Basic fields
        token = data.get("address", "")
        name = data.get("name")
        symbol = data.get("symbol")
        logo = data.get("logo")
        description = data.get("description")
        creation_time = parse_datetime(data.get("creationTime"))
        creation_block = data.get("creationBlock")
        decimals = data.get("decimals")

        # Social info might be nested in "socialInfo"
        social_info = data.get("socialInfo", {})

        # Audit fields (some might be missing if this dict is only "base" info)
        is_open_source = data.get("isOpenSource")
        is_honeypot = data.get("isHoneypot")
        is_mintable = data.get("isMintable")
        is_proxy = data.get("isProxy")
        slippage_modifiable = data.get("slippageModifiable")
        is_blacklisted = data.get("isBlacklisted")
        sell_tax = data.get("sellTax", {})
        buy_tax = data.get("buyTax", {})
        is_contract_renounced = data.get("isContractRenounced")
        is_potentially_scam = data.get("isPotentiallyScam")
        updated_at = parse_datetime(data.get("updatedAt"))

        # Decide if it's suspicious
        suspicious = (is_blacklisted == "yes" or is_potentially_scam == "yes")

        return cls(
            token=token,
            name=name,
            symbol=symbol,
            logo=logo,
            description=description,
            creation_time=creation_time,
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
            updated_at=updated_at,
            is_suspicious=suspicious
        )


from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DexToolsTokenDetails:
    token: str
    chain: str
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    mcap: Optional[float] = None
    fdv: Optional[float] = None
    holders: Optional[int] = None
    transactions: Optional[int] = None

    price: Optional[float] = None
    price_chain: Optional[float] = None
    price_5m: Optional[float] = None
    variation_5m: Optional[float] = None
    price_1h: Optional[float] = None
    variation_1h: Optional[float] = None
    price_6h: Optional[float] = None
    variation_6h: Optional[float] = None
    price_24h: Optional[float] = None
    variation_24h: Optional[float] = None

    @classmethod
    def from_dict(cls, address: str, chain: str, info_data: Dict, price_data: Dict) -> "DexToolsTokenDetails":
        """
        Create a DexToolsTokenDetails object from the given API responses.

        Args:
            address (str): The token address.
            chain (str): The blockchain chain (e.g., "solana", "ethereum").
            info_data (Dict): Data from the `/info` endpoint.
            price_data (Dict): Data from the `/price` endpoint.

        Returns:
            DexToolsTokenDetails: The populated dataclass instance.
        """
        # Extract data from the 'data' key in the responses
        info = info_data.get("data", {})
        price = price_data.get("data", {})

        return cls(
            token=address,
            chain=chain,
            circulating_supply=info.get("circulatingSupply"),
            total_supply=info.get("totalSupply"),
            mcap=info.get("mcap"),
            fdv=info.get("fdv"),
            holders=info.get("holders"),
            transactions=info.get("transactions"),
            price=price.get("price"),
            price_chain=price.get("priceChain"),
            price_5m=price.get("price5m"),
            variation_5m=price.get("variation5m"),
            price_1h=price.get("price1h"),
            variation_1h=price.get("variation1h"),
            price_6h=price.get("price6h"),
            variation_6h=price.get("variation6h"),
            price_24h=price.get("price24h"),
            variation_24h=price.get("variation24h"),
        )

