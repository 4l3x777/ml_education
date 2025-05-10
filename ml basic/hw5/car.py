from .base import Vehicle
from .engine import Engine

class Car(Vehicle):
    """
    Класс Car, наследник Vehicle.
    """
    def __init__(self, weight=0, fuel=0, fuel_consumption=0):
        super().__init__(weight, fuel, fuel_consumption)
        self.engine = None  # Атрибут для хранения двигателя

    def set_engine(self, engine: Engine):
        """
        Устанавливает двигатель для автомобиля.
        :param engine: экземпляр класса Engine
        """
        self.engine = engine
