
#import unittest
import matplotlib.pyplot as plt

import pybubble.models as bm

def test_Gilmore_ode():
    bm.Gilmore_ode(500e-6, 0, 50e-6, 0, 100, 0.1)

def test_GilmoreEick_ode():
    t, R, R_dot, pg, T = bm.GilmoreEick_ode(500e-6, 0, 50e-6, 0, 100, 0.01)

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t, R, '.')
    plt.ylabel('$R$ [m]')
    plt.subplot(4, 1, 2)
    plt.plot(t, R_dot, '.-')
    plt.ylabel('$\dot R$ [m/s]')
    plt.subplot(4, 1, 3)
    plt.plot(t, pg, '.-')
    plt.ylabel('$p_g$ [Pa]')
    plt.subplot(4, 1, 4)
    plt.plot(T, '.-')
    plt.ylabel('$T$ [K]')
    plt.show()

if __name__ == '__main__':
#    test_Gilmore_ode()
    test_GilmoreEick_ode()
    #    unittest.main()
