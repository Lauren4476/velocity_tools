'''
This file contains helper functions for the larger functions in velocity-tools.

TODO:
- Check these functions all work with jax. e.g. define constants as JAX scalars
'''

# Constants
G = 6.67430e-11 * (1e-3)**2 * (1.988416e30) / (1.4959787e11) # in au (km/s)^2 * Msol^-1
au_in_km = 1.4959787e8 #km

def au_to_m(quantity):
    result = quantity * (1.4959787e11)
    return result

def m_to_au(quantity):
    result = quantity / (1.4959787e11)
    return result

def solmass_to_kg(quantity):
    result = quantity * (1.988416e30)
    return result

def kg_to_solmass(quantity):
    result = quantity / (1.988416e30)
    return result

