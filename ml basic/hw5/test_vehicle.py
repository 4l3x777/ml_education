import pytest
from .car import Car
from .plane import Plane
from .engine import Engine
from .exceptions import LowFuelError, NotEnoughFuel, CargoOverload

def test_car_initialization():
    car = Car(weight=1500, fuel=50, fuel_consumption=5)
    assert car.weight == 1500
    assert car.fuel == 50
    assert car.fuel_consumption == 5
    assert car.engine is None

def test_car_set_engine():
    car = Car()
    engine = Engine(volume=2.0, pistons=4)
    car.set_engine(engine)
    assert car.engine == engine

def test_car_start():
    car = Car(fuel=10)
    car.start()
    assert car.started is True

    car_no_fuel = Car(fuel=0)
    with pytest.raises(LowFuelError):
        car_no_fuel.start()

def test_car_move():
    car = Car(fuel=20, fuel_consumption=5)
    car.move(3)
    assert car.fuel == 5

    with pytest.raises(NotEnoughFuel):
        car.move(2)

def test_plane_initialization():
    plane = Plane(weight=10000, fuel=500, fuel_consumption=50, max_cargo=2000)
    assert plane.weight == 10000
    assert plane.fuel == 500
    assert plane.fuel_consumption == 50
    assert plane.max_cargo == 2000
    assert plane.cargo == 0

def test_plane_load_cargo():
    plane = Plane(max_cargo=1000)
    plane.load_cargo(500)
    assert plane.cargo == 500

    with pytest.raises(CargoOverload):
        plane.load_cargo(600)

def test_plane_remove_all_cargo():
    plane = Plane(max_cargo=1000)
    plane.load_cargo(500)
    removed_cargo = plane.remove_all_cargo()
    assert removed_cargo == 500
    assert plane.cargo == 0

def test_plane_start_and_move():
    plane = Plane(fuel=1000, fuel_consumption=100)
    plane.start()
    assert plane.started is True

    plane.move(5)
    assert plane.fuel == 500

    with pytest.raises(NotEnoughFuel):
        plane.move(6)

def test_low_fuel_error():
    car = Car(fuel=0)
    with pytest.raises(LowFuelError, match="Недостаточно топлива для запуска."):
        car.start()

def test_not_enough_fuel_error():
    car = Car(fuel=10, fuel_consumption=5)
    with pytest.raises(NotEnoughFuel, match="Недостаточно топлива для преодоления расстояния."):
        car.move(3)

def test_cargo_overload_error():
    plane = Plane(max_cargo=1000)
    plane.load_cargo(800)
    with pytest.raises(CargoOverload, match="Превышена максимальная грузоподъемность!"):
        plane.load_cargo(300)