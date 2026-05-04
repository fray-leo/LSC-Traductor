"""
logic/database.py

Gestiona el historial de traducciones en una base de datos SQLite local.
Crea el archivo y la tabla automáticamente si no existen.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Clase principal
# ------------------------------------------------------------------

class TranslationDB:
    """
    Interfaz simple sobre SQLite para guardar y consultar
    el historial de señas detectadas.
    """

    CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS traducciones (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            sena      TEXT    NOT NULL,
            timestamp TEXT    NOT NULL
        )
    """

    def __init__(self, db_path: Path | str = None):
        """
        Args:
            db_path: Ruta al archivo .db. Si no se provee, se usa db/historial.db
                     en la raíz del proyecto.
        """
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent.parent / "db" / "historial.db"
        
        self._path = Path(db_path)
        
        # Asegurar que el directorio de la DB exista
        self._path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def save(self, sena: str) -> int:
        """
        Guarda una seña detectada con el timestamp actual.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO traducciones (sena, timestamp) VALUES (?, ?)",
                (sena, timestamp),
            )
            conn.commit()
            return int(cursor.lastrowid) if cursor.lastrowid is not None else 0
        finally:
            conn.close()

    def get_recent(self, n: int = 10) -> list[tuple]:
        """
        Devuelve las últimas n traducciones en orden cronológico inverso.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, sena, timestamp FROM traducciones ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
            return rows
        finally:
            conn.close()

    def get_all(self) -> list[tuple]:
        """Devuelve todas las traducciones en orden cronológico."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, sena, timestamp FROM traducciones ORDER BY id ASC"
            ).fetchall()
            return rows
        finally:
            conn.close()

    def count(self) -> int:
        """Devuelve el número total de traducciones guardadas."""
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM traducciones").fetchone()[0]
        finally:
            conn.close()

    def clear(self):
        """
        Elimina todas las traducciones del historial.
        """
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM traducciones")
                conn.execute("DELETE FROM sqlite_sequence WHERE name='traducciones'")
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _init_db(self):
        """Crea la tabla si no existe. Se llama una sola vez en __init__."""
        conn = self._connect()
        try:
            with conn:
                conn.execute(self.CREATE_TABLE)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        """
        Abre una conexión con check_same_thread=False para compatibilidad
        con el loop de tkinter, que corre en el hilo principal.
        """
        return sqlite3.connect(self._path, check_same_thread=False)