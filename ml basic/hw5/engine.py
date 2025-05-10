from dataclasses import dataclass

@dataclass
class Engine:
    """
    Дата-класс для описания двигателя.
    """
    volume: float  # Объем двигателя
    pistons: int   # Количество поршней
