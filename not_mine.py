from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt

J = 0.015
d = 0.089
r = 0.1
Kt = 0.026
Rr = 4.3
M = 1.5
l = 5
s3 = math.sqrt(3)
B = (l * Kt / (r * Rr)) * np.array([[-s3 / (2 * M), 0, s3 / (2 * M)],
                                    [1 / (2 * M), 1 / M, 1 / (2 * M)],
                                    [d / J, d / J, d / J]])

tt = np.linspace(0, 10, 200)
mu = -8
Vy = 1
Y0 = np.array([0, 0, 0, 0.3, 0, 0.1, 0, 0, 0, 0.3, 0, 0.1, 0, 0, 0])
A = np.array([[-0.48, 0, 0],
              [0, -0.53, 0],
              [0, 0, -0.023]])

K1 = np.eye(3, 3)
K2 = np.eye(3, 3)
Kd = np.array([[-3.8106, -0.0012, 0.5733],
               [-0.0000, 0.0083, 0.5727],
               [3.8106, -0.0012, 0.5733]])
Kv = np.array([[1.3608, -0.0000, -0.0000],
               [0.0000, 831.2001, -2.6982],
               [0.0000, -2.7928, 1.0165]])

# equilibrium theta
th0 = 0
R0 = np.array([[math.cos(th0), -math.sin(th0), 0],
               [math.sin(th0), math.cos(th0), 0],
               [0, 0, 1]]).T
# F(0)
F = np.linalg.inv(B) @ ((A - B @ Kd) @ R0 @ K2 - R0 @ K1);
# disturbance
h = np.array([0.1, 0.1, 0.1])


def dy(y, t):
    zv = y[0:3]
    zn = y[3:6]
    V = y[6:9]
    n = y[9:12]
    p = y[12:15]
    theta = n[2]
    eps = n - zn

    R = np.array([[math.cos(theta), -math.sin(theta), 0],
                  [math.sin(theta), math.cos(theta), 0],
                  [0, 0, 1]])
    e = n - np.array([0, Vy * t, 0])
    Vd = mu * e

    tau = -Kd @ (zv - Kv @ Vd) + p

    dzv = A @ zv + B @ tau + R.T @ K1 @ eps
    dzn = R @ zv + K2 @ eps
    dV = A @ V + B @ tau + h
    dn = R @ V
    dp = -p + F @ eps

    return np.concatenate((dzv, dzn, dV, dn, dp))
#
times = tt
# plt.subplot(211)
# plt.plot(tt, Y[:, 9])
# plt.plot([tt[0], tt[-1]], [0, 0])
#
# plt.subplot(212)
# plt.plot(tt, Y[:, 8])
# plt.show()
Y = odeint(dy, Y0, tt)

v   = np.array(Y[:,6])
vn  = np.array(Y[:,7])
w   = np.array(Y[:,8])
x   = np.array(Y[:,9])
y   = np.array(Y[:,10])
phi = np.array(Y[:,11])

fig, (xy, fi, speeds) = plt.subplots(nrows=3, ncols=1,
                             figsize = (10,6))
xy.set_title('y(x)')
xy.style = 'points'
xy.plot(x, y, color = 'black')
fi.set_title('phi(t)')
fi.plot(times, phi, color = 'black')
speeds.set_title('speeds/time: v: -.- ; vn: -- ; w: __ ')
speeds.plot(times, v, color = 'black', ls="-.", alpha=0.7)
speeds.plot(times, vn, color = 'black', ls='--', alpha=0.85)
speeds.plot(times, w, color = 'black', alpha=0.7)
plt.show()