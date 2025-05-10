"""
Объявите следующие исключения:
- LowFuelError
- NotEnoughFuel
- CargoOverload
"""

class LowFuelError(Exception):
    """Исключение: недостаточно топлива."""
    pass

class NotEnoughFuel(Exception):
    """Исключение: топлива недостаточно для завершения поездки."""
    pass

class CargoOverload(Exception):
    """Исключение: перегрузка груза."""
    pass
