import math
import scipy as sc
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import *
from scipy import integrate

pi = math.pi;
wheels = 3
angle = pi/3
theta = pi/wheels

d = 0.089
incoming = np.array([1, 1, 1])

Coeff = np.array(([-math.sin(angle), math.cos(angle), d],
                [0, -1, d],
                [math.sin(angle), math.cos(angle), d]))

Xr = (np.linalg.inv(Coeff)).dot(incoming)

v = Xr[0]; vn = Xr[1]; w = Xr[2]

l = 5   #gearbox reduction
r = 0.0325  #wheel radius
Kt = 0.0259 #motor torque constant
R = 4.3111  #resistor
M = 1.944   #mass
J = 0.0169  #inertia moment

"!!!!"
i = np.empty(wheels)   #motor current
for j in range(wheels):
    i[j] = 10

T = np.empty(wheels)
for j in range (wheels):
    T[j] = l*Kt*i[j]

f = np.zeros(wheels)
for j in range (wheels):
    f[j] = T[j]/r

def f0(t): return 0
def f1(t): return 0
def f2(t): return 0


def FBv(v):     return Bv * v
def FBvn(vn):   return Bvn * vn
def TBw(w):     return Bw * w
def FCv(v):     return Cv * np.sign(v)
def FCvn(vn):   return Cvn * np.sign(vn)
def TCw(w):     return Cw * np.sign(w)

def SumFv(t):   return (f2(t) - f0(t))*math.sin(angle);
def SumFvn(t):  return (f2(t) + f0(t))*math.cos(angle) - f[1];
def SumT(t):    return (f2(t) + f0(t) + f1(t))*d;

Bv = 0.5082; Bvn = 0.4870; Bw = 0.0130
Cv = 1.9068; Cvn = 2.0423; Cw = 0.0971

def dvdt(v, t):     return (SumFv(t) - FBv(v) - FCv(v))/M
def dvndt(vn, t):   return (SumFvn(t) - FBvn(vn) - FCvn(vn))/M
def dwdt(w, t):     return (SumT(t) - TBw(w) - TCw(w))/J

t = [0, 1]
Xo = np.empty((3,1))
Xo[0] = sc.integrate.odeint(dvdt, v, t)[0]
Xo[1] = sc.integrate.odeint(dvndt, vn, t)[0]
Xo[2] = sc.integrate.odeint(dwdt, w, t)[0]

Xr = np.array([Xr])
R_matrix = Xo.dot(Xr)

alpha = (l*Kt)/(r*R)
root = math.sqrt(3)
c = (-3)*alpha*alpha*R

A_11 = (c/(2*M) - Bv/M)
A_22 = (c/(2*M) - Bvn/M)
A_33 = ((c*d*d)/J - Bw/J)
A = np.array(([A_11, 0, 0],
              [0, A_22, 0],
              [0, 0, A_33]))

B = np.array(([(-root)/(2*M), 0, root/(2*M)],
              [1/(2*M), 1/M, 1/(2*M)],
              [d/J, d/J, d/J]))
B = alpha*B

K = np.array(([-Cv/M, 0, 0],
              [0, -Cvn/M, 0],
              [0, 0, -Cw/J]))

time = [0, 5]
sections = 5000
dt = (time[1]-time[0])/sections
times = np.empty(sections)
times[0] = time[0]
for j in range(sections):
    times[j] = times[0] + (dt*j)

def u0(t):  return 1
def u1(t):  return 1
def u2(t):  return 1
def U(t):   return np.array([u0(t), u1(t), u2(t)])

y0 = np.zeros((6,1))
y0 = y0[:,0]
print(y0)
def dydt(y, t):
    R_matrix = np.array(( [math.cos(y[5]), math.sin(y[5]), 0],
                        [-math.sin(y[5]), math.cos(y[5]), 0],
                        [0, 0, 1] ))
    R_matrix = np.linalg.inv(R_matrix)

    Etha = np.concatenate((A, R_matrix), axis=0)

    zeros = np.zeros((6, 3))
    Etha = np.concatenate((Etha, zeros), axis=1)

    zeros = np.zeros((3, 3))
    Betha = np.concatenate((B, zeros), axis=0)

    return Etha.dot(y) + Betha.dot(U(t));

Y = sc.integrate.odeint(dydt, y0, times)

v   = np.array(Y[:,0])
vn  = np.array(Y[:,1])
w   = np.array(Y[:,2])
x   = np.array(Y[:,3])
y   = np.array(Y[:,4])
phi = np.array(Y[:,5])
print(v, vn, w)
print(x)
print(y)
print(phi)
fig, (xy, fi, speeds) = plt.subplots(nrows=3, ncols=1,
                             figsize = (10,6))
xy.set_title('y(x)')
xy.style = 'points'
xy.plot(x, y, color='blue')
fi.set_title('phi(t)')
fi.plot(times, phi, color='green', ls='--')
speeds.set_title('speeds: v-purple; vn-orange; w-red /time')
speeds.plot(times, v, color='purple', alpha=0.7)
speeds.plot(times, vn, color='orange', ls='--', alpha=0.85)
speeds.plot(times, w, color='red', alpha=0.7)
plt.show()

