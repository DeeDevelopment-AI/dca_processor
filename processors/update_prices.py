# price_update.py
import logging
import sys
import asyncio
import aiohttp
from db.connection import DatabaseConnection
from db.queries import DatabaseQueries
from config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('price_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def update_token_prices():
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()

        # Initialize database connection and queries
        logger.info("Initializing database connection...")
        db_connection = DatabaseConnection(config.db)
        db_queries = DatabaseQueries(db_connection)

        try:
            # Get active tokens
            logger.info("Fetching active tokens...")
            active_tokens = db_queries.get_active_tokens()
            logger.info(f"Found {len(active_tokens)} active tokens")

            if not active_tokens:
                logger.info("No active tokens found.")
                return

            # Process tokens with rate limiting
            async with aiohttp.ClientSession() as session:
                semaphore = asyncio.Semaphore(config.birdeye.max_concurrent_requests)
                retry_tokens = []
                processed_count = 0

                async def process_token(token_address: str):
                    nonlocal processed_count
                    async with semaphore:
                        try:
                            success = await db_queries.fetch_and_store_birdeye_prices(
                                session, token_address, config.birdeye
                            )

                            if success:
                                processed_count += 1
                                logger.info(f"Progress: {processed_count}/{len(active_tokens)}")
                            else:
                                retry_tokens.append(token_address)

                            # Sleep between requests
                            await asyncio.sleep(config.birdeye.rate_limit)

                        except Exception as e:
                            logger.error(f"Error processing {token_address}: {str(e)}")
                            retry_tokens.append(token_address)
                            if "rate limit" in str(e).lower():
                                logger.info("Rate limit hit, sleeping for 60 seconds...")
                                await asyncio.sleep(1)

                # Process in smaller batches
                batch_size = 1
                for i in range(0, len(active_tokens), batch_size):
                    batch = active_tokens[i:i + batch_size]
                    tasks = [process_token(token) for token in batch]
                    await asyncio.gather(*tasks)

                    # After each batch, sleep a bit
                    await asyncio.sleep(1)

                # Log results
                logger.info(f"Successfully processed {processed_count} tokens")
                if retry_tokens:
                    logger.info(f"{len(retry_tokens)} tokens need retry")
                    # Save retry tokens to file for later
                    with open('retry_tokens.txt', 'w') as f:
                        for token in retry_tokens:
                            f.write(f"{token}\n")

        except Exception as e:
            logger.error(f"An error occurred during processing: {str(e)}")
            raise

        finally:
            logger.info("Closing database connection...")
            db_connection.close()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(update_token_prices())