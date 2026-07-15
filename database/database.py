import mysql.connector
from config.config import DB_CONFIG


class Database:
    """
    Kelola 1 koneksi MySQL yang dipakai bareng-bareng
    sama semua repository (biar gak konek berkali-kali).
    """

    _connection = None

    @classmethod
    def get_connection(cls):
        if cls._connection is None or not cls._connection.is_connected():
            cls._connection = mysql.connector.connect(**DB_CONFIG)
        return cls._connection

    @classmethod
    def close_connection(cls):
        if cls._connection is not None and cls._connection.is_connected():
            cls._connection.close()
            cls._connection = None
