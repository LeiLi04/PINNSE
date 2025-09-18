def greet(name: str) -> str:
    """Return a friendly greeting for the provided name."""
    return f"Hello, {name}!"


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return celsius * 9.0 / 5.0 + 32.0


def kinematics_distance(v0: float, t: float, a: float = 0.0) -> float:
    """Compute displacement under constant acceleration.

    Uses s = v0 * t + 0.5 * a * t^2.
    Parameters are in SI units: meters, seconds, m/s^2.
    """
    return v0 * t + 0.5 * a * (t ** 2)


def test_celsius_to_fahrenheit():
    # Freezing point
    assert celsius_to_fahrenheit(0) == 32.0
    # Boiling point
    assert celsius_to_fahrenheit(100) == 212.0
    # A negative value
    assert celsius_to_fahrenheit(-40) == -40.0


def test_kinematics_distance():
    # No acceleration: s = v0 * t
    assert kinematics_distance(10.0, 5.0, 0.0) == 50.0
    # With acceleration: v0=0, a=2, t=3 -> s = 0.5*2*9 = 9
    assert kinematics_distance(0.0, 3.0, 2.0) == 9.0
    # Negative acceleration (deceleration)
    assert kinematics_distance(20.0, 2.0, -5.0) == 30.0


if __name__ == "__main__":
    # Example manual run
    print(celsius_to_fahrenheit(100))
    # y = kinematics_distance(1, 1, 1)
    # print(y)
    

y = kinematics_distance(1, 1, 1)
print(y)
