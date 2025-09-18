def greet(name: str) -> str:
    """Return a friendly greeting for the provided name."""
    return f"Hello, {name}!"


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return celsius * 9.0 / 5.0 + 32.0

def test_celsius_to_fahrenheit():
    # Freezing point
    assert celsius_to_fahrenheit(0) == 32.0
    # Boiling point
    assert celsius_to_fahrenheit(100) == 212.0
    # A negative value
    assert celsius_to_fahrenheit(-40) == -40.0

if __name__ == "__main__":
    # Example manual run
    print(celsius_to_fahrenheit(100))
