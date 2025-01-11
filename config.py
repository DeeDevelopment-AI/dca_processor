import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str


@dataclass
class QuicknodeConfig:
    api_url: str
    api_key: str


@dataclass
class DexToolsConfig:
    api_url: str
    api_key: str
    rate_limit: int = 300


@dataclass
class Config:
    db: DatabaseConfig
    quicknode: QuicknodeConfig
    dextools: DexToolsConfig
    log_level: str = "INFO"
    batch_size: int = 1


def load_config() -> Config:
    """Load configuration from environment variables"""
    return Config(
        db=DatabaseConfig(
            host=os.getenv('DB_HOST', 'dbaas-db-10089787-do-user-8166025-0.m.db.ondigitalocean.com'),
            port=int(os.getenv('DB_PORT', '25060')),
            database=os.getenv('DB_NAME', 'solana'),
            user=os.getenv('DB_USER', 'doadmin'),
            password=os.getenv('DB_PASSWORD', 'AVNS_oMROZR_lgDXiayEzYN7')
        ),
        quicknode=QuicknodeConfig(
            api_url=os.getenv('QUICKNODE_URL', 'https://yolo-sly-fire.solana-mainnet.quiknode.pro/ea87b760743f2193b31af6054b169e5a2383ad49'),
            api_key=os.getenv('QUICKNODE_API_KEY', 'QN_47906604a722460ebe28c087d8243566')
        ),
        dextools=DexToolsConfig(
            api_url=os.getenv('DEXTOOLS_API_URL', 'https://public-api.dextools.io/trial'),
            api_key=os.getenv('DEXTOOLS_API_KEY', 'YnPqmXSgoWafyeBqJT6oa1xfBjHVugQM4lzfP2pE'),
            rate_limit=int(os.getenv('DEXTOOLS_RATE_LIMIT', '1'))
        ),
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        batch_size=int(os.getenv('BATCH_SIZE', '1'))
    )