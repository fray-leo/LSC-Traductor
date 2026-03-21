"""
logic/database.py

Gestiona el historial de traducciones en una base de datos SQLite local.
Crea el archivo y la tabla automáticamente si no existen.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Ruta
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "historial.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Clase principal
# ------------------------------------------------------------------

class TranslationDB:
    """
    Interfaz simple sobre SQLite para guardar y consultar
    el historial de señas detectadas.

    Uso:
        db = TranslationDB()
        db.save("hola")
        recientes = db.get_recent(10)  # [(id, seña, timestamp), ...]
        db.clear()
    """

    CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS traducciones (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            sena      TEXT    NOT NULL,
            timestamp TEXT    NOT NULL
        )
    """

    def __init__(self, db_path: Path = DB_PATH):
        self._path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def save(self, sena: str) -> int:
        """
        Guarda una seña detectada con el timestamp actual.

        Args:
            sena: nombre de la seña (ej. "hola").

        Returns:
            ID de la fila insertada.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO traducciones (sena, timestamp) VALUES (?, ?)",
                (sena, timestamp),
            )
            return cursor.lastrowid

    def get_recent(self, n: int = 10) -> list[tuple]:
        """
        Devuelve las últimas n traducciones en orden cronológico inverso.

        Returns:
            Lista de tuplas (id, sena, timestamp).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, sena, timestamp FROM traducciones ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        return rows

    def get_all(self) -> list[tuple]:
        """Devuelve todas las traducciones en orden cronológico."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, sena, timestamp FROM traducciones ORDER BY id ASC"
            ).fetchall()
        return rows

    def count(self) -> int:
        """Devuelve el número total de traducciones guardadas."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM traducciones").fetchone()[0]

    def clear(self):
        """
        Elimina todas las traducciones del historial.
        Útil para limpiar antes de una demo.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM traducciones")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='traducciones'")

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _init_db(self):
        """Crea la tabla si no existe. Se llama una sola vez en __init__."""
        with self._connect() as conn:
            conn.execute(self.CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        """
        Abre una conexión con check_same_thread=False para compatibilidad
        con el loop de tkinter, que corre en el hilo principal.
        """
        return sqlite3.connect(self._path, check_same_thread=False)