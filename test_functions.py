import numpy as np
#TODO dodati ranac?

def ZDT1(x):
    n = len(x)  # Number of decision variables
    f1 = x[0]  # Objective 1 is just the first variable
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - np.sqrt(f1 / g))  # Objective 2 formula
    return f1, f2


def ZDT2(x):
    n = len(x)  # Number of decision variables
    f1 = x[0]  # Objective 1 is just the first variable
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - (f1 / g) ** 2)  # Objective 2 formula
    return f1, f2

def ZDT3(x):
    n = len(x)  # Number of decision variables
    f1 = x[0]  # Objective 1 is just the first variable
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)  # g function
    h =  1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1) # h function
    f2 = g * h  # Objective 2 formula
    return f1, f2

def ZDT4(x):
    n = len(x)  # Number of decision variables
    f1 = x[0]  # Objective 1 is just the first variable
    g = 1 + 10 * (n - 1) + np.sum(x[1:]**2 - 10 * np.cos(4 * np.pi * x[1:]))  # g function
    h =  1 - np.sqrt(f1 / g) # h function
    f2 = g * h  # Objective 2 formula
    return f1, f2

def ZDT6(x):
    n = len(x)  # Number of decision variables
    f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0])**6 # Objective 1 formula
    g = 1 + 9 * (np.sum(x[1:]) / (n - 1))**0.25  # g function
    h =  1 - (f1 / g)**2 # h function
    f2 = g * h  # Objective 2 formula
    return f1, f2
