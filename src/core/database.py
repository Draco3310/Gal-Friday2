import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseHandler:
    def __init__(self, db_path: str = "gal_friday.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize the database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Positions Table (Parent)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL, -- OPEN, PARTIAL, CLOSED
                    remaining_qty REAL NOT NULL,
                    current_stop_loss REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Executions Table (Child)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY,
                    position_id TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    reason TEXT NOT NULL, -- ENTRY, TP1, STOP_LOSS
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (position_id) REFERENCES positions (id)
                )
            """)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def create_position(self, position_id: str, symbol: str, qty: float, stop_loss: Optional[float] = None):
        """Creates a new parent position."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO positions (id, symbol, status, remaining_qty, current_stop_loss) VALUES (?, ?, ?, ?, ?)",
                (position_id, symbol, "OPEN", qty, stop_loss)
            )
            conn.commit()
            logger.info(f"Created position {position_id} for {symbol}")

    def add_execution(self, execution_id: str, position_id: str, qty: float, price: float, reason: str):
        """Adds a child execution record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO executions (id, position_id, qty, price, reason) VALUES (?, ?, ?, ?, ?)",
                (execution_id, position_id, qty, price, reason)
            )
            conn.commit()
            logger.info(f"Added execution {execution_id} to position {position_id}")

    def update_position(self, position_id: str, remaining_qty: Optional[float] = None, status: Optional[str] = None, stop_loss: Optional[float] = None):
        """Updates position details."""
        updates = []
        params = []

        if remaining_qty is not None:
            updates.append("remaining_qty = ?")
            params.append(remaining_qty)
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if stop_loss is not None:
            updates.append("current_stop_loss = ?")
            params.append(stop_loss)

        if not updates:
            return

        updates.append("updated_at = CURRENT_TIMESTAMP")
        query = f"UPDATE positions SET {', '.join(updates)} WHERE id = ?"
        params.append(position_id)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            logger.info(f"Updated position {position_id}")

    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM positions WHERE status IN ('OPEN', 'PARTIAL')"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
