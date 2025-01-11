import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Iterator, Any
from config import DatabaseConfig

class DatabaseConnection:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._conn = None

    @contextmanager
    def get_cursor(self) -> Iterator[Any]:
        """Get database cursor with automatic cleanup"""
        if not self._conn:
            self._conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )

        try:
            cursor = self._conn.cursor(cursor_factory=RealDictCursor)
            yield cursor
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise e
        finally:
            cursor.close()

    def call_procedures(self):
        """Call stored procedures"""
        with self.get_cursor() as cur:
            cur.execute("CALL process_hourly_dca_trends();")
            cur.execute("CALL process_hourly_dca_endandclose_trends();")

    def close(self):
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
