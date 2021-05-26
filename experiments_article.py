import math
import scipy as sc
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from scipy import *
from scipy import integrate

pi = math.pi

sections = 500
times = np.linspace(0, 15, sections)


A = -np.array(([0.48, 0, 0],
              [0, 0.53, 0],
              [0, 0, 0.023]))

B = np.array(([-0.1745, 0, 0.1745],
              [0.1008, 0.2016, 0.1008],
              [1.7938, 1.7938, 1.7938]))

K1 = 100*np.eye(3, 3)
K2 = 10*np.eye(3, 3)

Kd = np.array(([-3.81, -0.0012, 0.5733],
               [0, 0.0083, 0.5727],
               [3.8106, -0.0012, 0.5733]))

Kv = np.array(([1.3608, 0, 0],
               [0, 831.2, -2.6982],
               [0, -2.7928, 1.0165]))
mu = 8


def R_matrix(fi):
    return np.array(([math.cos(fi), -math.sin(fi), 0],
                     [math.sin(fi), math.cos(fi), 0],
                     [0, 0, 1]))


v = np.array([0, 0, 0])
etha0 = np.array([0, 0, 0])

inv_B = np.linalg.inv(B)
Rt = R_matrix(etha0[2]).transpose()
part1 = ((A - B @ Kd) @ Rt) @ K2
part2 = Rt @ K1
F0 = inv_B @ (part1 - part2)

Ap = -np.eye(3,3)
Bp = F0
Cp = np.eye(3, 3)
inv_Ap = np.linalg.inv(-Ap)
Fs = (Cp @ inv_Ap) @ Bp
Dp = np.zeros((3, 3))

def vd0(t): return np.array([0, 0.1*t, 0])
zv = np.zeros((3,))
zetha = np.zeros((3,))
p = np.zeros((3,))
eps = etha0 - zetha

q0 = np.concatenate((v, etha0, zv, zetha, p), axis=0)
#disturbance
h = np.array([0.1,0.1,0.1])

def dZ(q, t):
    to_concat = []
    R = R_matrix(q[5])
    inv_R = np.linalg.inv(R)

    to_concat.append(np.concatenate((A, R, np.zeros((9, 3)))))
    to_concat.append(np.zeros((15, 3)))
    to_concat.append(np.concatenate((np.zeros((6, 3)), A, R, np.zeros((3, 3)))))
    to_concat.append(np.zeros((15, 3)))
    to_concat.append(np.concatenate((np.zeros((12, 3)), Ap)))

    Q = np.concatenate(to_concat, axis=1)

    Etha = np.concatenate((np.zeros((6,3)), (R.T) @ K1, K2, Bp), axis=0)

    Betha = np.concatenate((B, np.zeros((3,3)), B, np.zeros((6,3))), axis=0)

    Ash = np.concatenate((h, np.zeros(12,)), axis=0)

    etha = q[3:6]
    e = (etha - vd0(t))
    vd = -mu*R@(e.T)
    Eps = etha - q[9:12]
    p = q[12:15]
    zv = q[6:9]
    ksi = Cp @ p + Dp @ Eps
    tau = -Kd @ (zv - Kv @ vd) + ksi
    dq = Q.dot(q) + Etha @ Eps + Betha @ tau + Ash
    return dq

Y = sc.integrate.odeint(dZ, q0, times)

v   = np.array(Y[:,0])
vn  = np.array(Y[:,1])
w   = np.array(Y[:,2])
x   = np.array(Y[:,3])
y   = np.array(Y[:,4])
phi = np.array(Y[:,5])
#
# forplot1 = np.concatenate((Y[:, 0:3], Y[:, 6:12]), axis=1)
# forplot2 = np.concatenate((Y[:, 0:2], Y[:, 6:12]), axis=1)
# plt.subplot(211)
# plt.plot(times, forplot1)
# plt.plot([times[0], times[-1]], [0, 0])
#
# plt.subplot(212)
# plt.plot(times, forplot2)
# plt.show()


fig, (xy, fi, speeds) = plt.subplots(nrows=3, ncols=1,
                                           figsize = (10,6))
# xy.set_title('y(x)')
#xy.axis([-0.015, 0.1, -0.015, 1.5])
xy.set_xlabel('Oy')
xy.set_ylabel('Ox')
xy.plot(y, x, color = 'black')

# fi.set_title('phi(t)')
fi.set_xlabel('time')
fi.set_ylabel('phi')
fi.plot(times, phi, color = 'black')

speeds.set_title('v: -.- ; vn: -- ; w: __ ')
speeds.set_xlabel('time')
speeds.set_ylabel('speeds')
speeds.plot(times, v, color = 'black', ls="-.", alpha=0.7)
speeds.plot(times, vn, color = 'black', ls='--', alpha=0.85)
speeds.plot(times, w, color = 'black', alpha=0.7)
plt.show()
