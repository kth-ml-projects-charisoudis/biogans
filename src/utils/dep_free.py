import math

from IPython import get_ipython


def closest_pow(x: int or float, of: int = 2) -> int or float:
    """
    Compute closest power of :attr:`of` for given number :attr:`x`
    :param (int or float) x: input number
    :param int of: exponent
    :return: the same data type as :attr:`x`
    """
    return type(x)(pow(of, round(math.log(x) / math.log(of))))


def in_notebook() -> bool:
    """
    Checks if code is run from inside
    :return: a bool object
    """
    try:
        shell_class = get_ipython().__class__
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or 'google.colab' in str(shell_class):
            return True  # Jupyter notebook or Qt console
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter]
