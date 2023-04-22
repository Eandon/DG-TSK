"""
membership functions used in the fuzzy system
"""


def gauss(x, m, sigma):
    """
    gaussian membership function
    exp(-(x-m)^2/(2*sigma^2))
    :param x: independent variable
    :param m: center
    :param sigma: spread
    :return: membership values
    """
    return (-(x - m) ** 2 / (2 * sigma ** 2)).exp()


def sim_gauss(x, m):
    """
    simplified gaussian membership function
    exp(-(x-m)^2)
    :param x: independent variable
    :param m: center
    :return: membership values
    """
    return (-(x - m) ** 2).exp()
