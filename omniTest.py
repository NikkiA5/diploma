from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

#ArUco marker draw on the blank bg image

marker_size = 200
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
marker_image = cv2.aruco.drawMarker(dictionary, 0, marker_size);
img_center = tuple((np.array(marker_image.shape)-1) / 2)
rot_max_side = int(round(marker_size*math.sqrt(2)))
rot_size = (rot_max_side, rot_max_side)
rot_shift = rot_max_side/2.0 - marker_size/2.0
big_img_size = (600, 600)
scale = 1.0

def draw_marker(bg_img,x,y,phi):
	bg_img = 255*np.ones(big_img_size, dtype=np.uint8)
	rot_mat = cv2.getRotationMatrix2D(img_center, phi, scale)
	rot_mat[1,2] += rot_shift   #alpha+
	rot_mat[0,2] += rot_shift   #betha+
	result = cv2.warpAffine(marker_image, rot_mat, rot_size, flags=cv2.INTER_LINEAR, borderValue=255)
	cnt_x = int(round(x - rot_max_side/2))
	cnt_y = int(round(y - rot_max_side/2))
	bg_img[cnt_x:cnt_x+rot_max_side,cnt_y:cnt_y+rot_max_side] = result
	return bg_img

#Two-Step modelling

J = 0.015
d = 0.089
r = 0.1
Kt = 0.026
Rr = 4.3
M = 1.5
l = 5
s3 = math.sqrt(3)
B = (l*Kt/(r*Rr))*np.array([[-s3/(2*M), 0, s3/(2*M)],
                            [1/(2*M), 1/M, 1/(2*M)],
                            [d/J, d/J, d/J]])

tt = np.linspace(0,10,200)
mu = -8
Vy = 1
Y0 = np.array([0,0,0,0.3,0,0.1])
Z0 = np.array([0,0,0,0.3,0,0.1,0,0,0])
A = np.array([[-0.48,0,0],
              [0,-0.53,0],
              [0,0,-0.023]])

K1 = np.eye(3,3)
K2 = np.eye(3,3)
Kd = np.array([[-3.8106, -0.0012, 0.5733],
			   [-0.0000, 0.0083, 0.5727],
			   [3.8106, -0.0012, 0.5733]])
Kv = np.array([[1.3608, -0.0000, -0.0000],
			   [0.0000, 831.2001, -2.6982],
			   [0.0000, -2.7928, 1.0165]])

#equilibrium theta
th0 = 0
R0 = np.array([[math.cos(th0), -math.sin(th0),0],
               [math.sin(th0), math.cos(th0),0],
               [0,0,1]]).T
#F(0)
F = np.linalg.inv(B)@((A-B@Kd)@R0@K2 - R0@K1)
#disturbance
h = np.array([0.1,0.1,0.1])
dt = 0.05
T = 10
npoints = 201
tau0 = np.array([0.0, 0.0, 0.0])
t_int = np.array([0, dt])
tt = np.linspace(0,T+dt,npoints)
y_log = Y0
t = 0

# Robot dynamics right part
def dy(y, t, tau):
    V = y[0:3]
    n = y[3:6]
    theta = n[2]

    R = np.array([[math.cos(theta), -math.sin(theta),0],
				  [math.sin(theta), math.cos(theta),0],
				  [0,0,1]])

    dV   = A @ V  + B @ tau + 0*h
    dn   = R @ V

    return np.concatenate((dV,dn))

# Regulator dynamics right part
def dz(y, t, n, tau):
    zv = y[0:3]
    zn = y[3:6]
    p = y[6:9]
    theta = n[2]
    eps = n - zn

    R = np.array([[math.cos(theta), -math.sin(theta),0],
				  [math.sin(theta), math.cos(theta),0],
				  [0,0,1]])

    dzv  = A @ zv + B @ tau + R.T @ K1 @ eps
    dzn  = R @ zv + K2 @ eps
    dp   = -p + F@eps

    return np.concatenate((dzv,dzn,dp))

state_x, state_y, state_phi = 300, 300, 0
# Iterate through all time points
for i in range(0,npoints-1):
    # Integrate dynamics
    Y = odeint(dy, Y0, t_int, args=(tau0, ))
    Y0 = Y[-1,:]
    n = Y0[3:6]
    t += dt
    y_log = np.vstack((y_log,Y0))

    state_x = 300 + n[0]
    state_y = 300 - n[1]
    state_phi += n[2]*180/math.pi

    # !!! Here goes ArUco marker draw+detection !!!
    marker = 255 * np.ones(big_img_size, dtype=np.uint8)
    marker = draw_marker(marker,state_x, state_y, state_phi)

    parameters = cv2.aruco.DetectorParameters_create()
    corners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(marker, dictionary, parameters=parameters)

    x = (corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]) / 4
    y = (corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]) / 4

    angle = math.acos((x-300)/math.sqrt((x-300)**2 + (y-300)**2))
    degrees = angle*180/math.pi
    # angle = degrees*math.pi/180

    n[0] = x - 300
    n[1] = y - 300
    n[2] = angle
    #print(n, degrees)

    n_estim = n #this should be replaced by estimate from detection
    # Integrate regulator
    Z = odeint(dz, Z0, t_int, args=(n_estim, tau0))
    Z0 = Z[-1,:]
    zv = Z0[0:3]
    p = Z0[6:9]
    e = n - np.array([0,Vy*t,0])
    Vd = mu*e
    # Get control signal
    tau0  = -Kd @ (zv - Kv@Vd) + p

plt.subplot(211)
#plot x coord
plt.plot(tt, y_log[:,3])
#target line
plt.plot([tt[0], tt[-1]],[0, 0])

plt.subplot(212)
plt.plot(tt, y_log[:,2])
plt.show()



bg_img = 255*np.ones(big_img_size, dtype=np.uint8)
bg_img = draw_marker(bg_img,300,300,30)
cv2.imshow("aruco", bg_img)
cv2.waitKey(0)
cv2.destroyAllWindows()