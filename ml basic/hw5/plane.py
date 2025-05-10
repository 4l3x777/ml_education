from .base import Vehicle
from .exceptions import CargoOverload

class Plane(Vehicle):
    """
    Класс Plane, наследник Vehicle.
    """
    def __init__(self, weight=0, fuel=0, fuel_consumption=0, max_cargo=0):
        """
        Инициализация самолета.
        :param weight: вес самолета
        :param fuel: количество топлива
        :param fuel_consumption: расход топлива
        :param max_cargo: максимальная грузоподъемность
        """
        super().__init__(weight, fuel, fuel_consumption)
        self.cargo = 0
        self.max_cargo = max_cargo

    def load_cargo(self, cargo_weight):
        """
        Загружает груз на самолет.
        :param cargo_weight: вес груза
        :raises CargoOverload: если груз превышает максимальную грузоподъемность
        """
        if self.cargo + cargo_weight > self.max_cargo:
            raise CargoOverload("Превышена максимальная грузоподъемность!")
        self.cargo += cargo_weight

    def remove_all_cargo(self):
        """
        Убирает весь груз с самолета.
        :return: вес груза, который был до обнуления
        """
        removed_cargo = self.cargo
        self.cargo = 0
        return removed_cargo
