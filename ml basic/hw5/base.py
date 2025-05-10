from abc import ABC
from .exceptions import LowFuelError, NotEnoughFuel

class Vehicle(ABC):
    def __init__(self, weight=0, fuel=0, fuel_consumption=0):
        """
        Инициализация базового класса Vehicle.
        :param weight: вес транспортного средства
        :param fuel: количество топлива
        :param fuel_consumption: расход топлива на единицу расстояния
        """
        self.weight = weight
        self.started = False
        self.fuel = fuel
        self.fuel_consumption = fuel_consumption

    def start(self):
        """
        Запускает транспортное средство.
        Если топлива нет, выбрасывает исключение LowFuelError.
        """
        if not self.started:
            if self.fuel > 0:
                self.started = True
            else:
                raise LowFuelError("Недостаточно топлива для запуска.")

    def move(self, distance):
        """
        Перемещает транспортное средство на указанное расстояние.
        Если топлива недостаточно, выбрасывает исключение NotEnoughFuel.
        :param distance: расстояние, которое нужно преодолеть
        """
        required_fuel = distance * self.fuel_consumption
        if self.fuel >= required_fuel:
            self.fuel -= required_fuel
        else:
            raise NotEnoughFuel("Недостаточно топлива для преодоления расстояния.")
