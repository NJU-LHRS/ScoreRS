class CustomTrainerError(Exception):
    """
    Base class for all custom OLMo exceptions.
    """


class CustomTrainerConfigError(CustomTrainerError):
    """
    An error with a configuration file.
    """
