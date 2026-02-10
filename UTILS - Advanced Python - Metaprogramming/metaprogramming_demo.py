"""
Demonstration of Python Metaprogramming concepts.
"""


class Singleton(type):
    """
    A metaclass for creating Singleton classes.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DatabaseConnection(metaclass=Singleton):
    """
    A class that can only have one instance.
    """

    def __init__(self):
        print("Initializing Database Connection...")


def create_class_dynamically(name):
    """
    Demonstrates using type() to create a class at runtime.
    """

    def say_hello(self):
        return f"Hello from {self.__class__.__name__}"

    return type(name, (object,), {"greet": say_hello})


if __name__ == "__main__":
    print("Metaprogramming - Singleton Pattern")
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    print(f"Is db1 the same as db2? {db1 is db2}")

    print("\nDynamic Class Creation")
    DynamicQuant = create_class_dynamically("DynamicQuant")
    dq = DynamicQuant()
    print(dq.greet())
