#Attempt to calculate altitude, velocity and dynamic pressure of a rocket in an interative system

# This code uses the thrust force T, air resistance Fd and gravity to calculate the trajectory over a given time of a rocket.
# For the Air resistance I used the atmosphereic model from NASA https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/earth-atmosphere-equation-metric/
# The used drac coeefcient is that of a cone. I dont know ours yet nor do i know how i evolves over density and velocity changes.
# This model has no calculation for lift forces
# The coordinate system is a x-y-z System but we only look at the x-z Plane. x horizontal, z vertical
# mx_k is a matrix with k1 and k1 as factors on it diagonal. they are trig-functions. at the moment you can plug in an angle of 0° - 90°
# 0° would mean the rocket starts vertical, and 90° would mean a horizontal start, wich obv. makes no sense.
# in the mx_k def is teh facto f which will be a function in the future to make the rocket start vertically and then let it change its angle over time
# the base formual is a*m = T - m*g - Fd which is a differential equation. This skript is an attempt to bypass the need of solving this equation
# m is a constat function of time. so we have a = T/m - g - Fd/m . a*dt with dt beeing really small we get a velocity, wich we add on our initial velocity
# so we have a1 wich leads to dv1 so we have v0 + dv1 = v1. v1*dt = dh1 and h0 + dh1 = h1. a1, v1 and h1 are the new values for the next calulation, wich will-
# -result in a2. a2 will result in v2 and h2 and the cycle repeats. and the smaller dt is, the more precice the calculation gets.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   Formulas
#------------------------------------------------------------------------------------------------------------------------------------------------

def A(r):                           # crossection area of the rocket
    return PI * r**2

def mt(m, U, t):                    # the mass over time. wett mass minus fuel consumption multiplyed by time
    return m - U * t

def d(h):                           # air (d)ensity
    if h < 0:
        raise ValueError("error: negative h-value")
        
    elif 0 <= h < 11000:            # Troposphere
        T = 15.04 - 0.00649 * h +273.1
        p = 101.29 * (T / 288.08) ** 5.256
        
    elif 11000 <= h < 25000:        # Lower Stratosphere
        T = -56.46 + 273.1
        p = 22.65 * math.exp(1.73 - 0.000157 * h)
        
    elif h >= 25000:                # Upper Stratosphere
        T = -131.21 + 0.00299 * h + 273.1
        p = 2.488 * (T / 216.6) ** (-11.388)
        
    else:
        raise ValueError("error: h-value out of range")
        
    d = p / (0.2869 * T)  # density in kg/m^3
    return d

def q(d, v):                        # dynamic pressure in Pa = N/m^2
    return 0.5 * d * v**2

def Fd(q, Cd, A):                   # Air resistance. F = 1/2 * Cd * A * d(density) * v^2. Because 1/2 * d * v^2 is q we can substitude it
    return q * A * Cd

def T(t):                           # Thrust over time. Burnout after 76s after Launch
    if t<=76:
        T = 28000
    else:
        T= 0
    return T

def g(h):                           # graviataion factor/ the big number is a factor. Mass of earth and G. The constant is choosen so at h_0, g(h) = g_0
    g = 398184378.21 / (6371 + (h/1000))**2
    return g

def mx_k(t):
    f = 0*t + 0.01745329*1
    k1 = np.cos(PI/2 - f)
    k2 = np.sin(PI/2 - f)
    k = np.array([[k1, 0], [0, k2]])
    return k

def sgn(x):
    if x == 0:
        s = 0
    elif x > 0:
        s = 1
    elif x < 0:
        s = -1
    else:
        raise ValueError("error: issue with sgn-function")
    return s

#------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   Values
#------------------------------------------------------------------------------------------------------------------------------------------------

Cd = 0.5                # no unit
R = 0.225               # m
t_0 = 0                 # s
v_0 = 0                 # m/s
h_0 = 0
x_0 = 0                 # m
t_e = 76                # s
Thrust_const = 28000    # N
m_w = 1000              # kg
U = 9.32454             # kg/s
g_0 = 9.81              # m/s^2
c = 331                 # m/s
PI = round(np.pi, 6)    

#------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                  Calculation
#------------------------------------------------------------------------------------------------------------------------------------------------

altitude = []
distance = []
dynamic_pressure = []
drag = []
time = []
velocity = []
mach = []
thrust = []
geo_pot = []

t = t_0
vec_xh = np.array([[x_0], [h_0]])
vx = v_0
vz = v_0
v_vector = np.array([[vx], [vz]])
dt = 0.001
mx_dt = np.array([[dt, 0], [0, dt]])
dv = 0
mx_sgn = np.array([[sgn(dv), 0], [0, sgn(dv)]])

t_max = 750

while t < t_max:
    
    #base values
    x = vec_xh[0, 0]
    h = vec_xh[1, 0]
    v = (v_vector[0, 0]**2 + v_vector[1, 0]**2)**0.5
    
    #check
    if t > 0.01*t_max:
        if h <= 0:
            break
                    
    mh = v/c
    m_t = mt(m_w, U, t)
    T_t = T(t)
    q_t = q(d(h), v)
    Fd_t= Fd(q_t, Cd, A(R))
    
    #accelerations
    g_t = g(h)
    a_th = T_t/m_t
    a_Fd = Fd_t/m_t
    
    #lists for evaluation
    time.append(t)
    thrust.append(T_t)
    altitude.append(round(h, 5))
    distance.append(round(x, 5))
    velocity.append(round(v, 5))
    mach.append(round(mh, 3))
    dynamic_pressure.append(round(q_t, 5))
    drag.append(round(Fd_t, 5))
    geo_pot.append(round(g_t, 5))
    
    #vectors
    vec_ath = np.array([[a_th], [a_th]])
    
    vec_g = np.array([[0], [g_t]])
    
    vec_aFd = np.array([[a_Fd], [a_Fd]])
    
    vec_dv = mx_dt @ (mx_k(t) @ (vec_ath - mx_sgn @ vec_aFd) - vec_g)
    
    dv = vec_dv[1, 0]
    
    v_vector = v_vector + vec_dv
    
    vec_dxdh = mx_dt @ v_vector
    
    vec_xh = vec_xh + vec_dxdh
    
    #time iteration
    t = t + dt
        
#------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   Evaluation
#------------------------------------------------------------------------------------------------------------------------------------------------

#generates a tabula which shows all calculated values at a whole sec (a whole sec is 1s, 2s, 3s and not 2.001s or 3.278s)
df = pd.DataFrame({
    'time': time,
    'thrust': thrust,
    'h': altitude,
    'v': velocity,
    'mach': mach,
    'g': geo_pot,
    'q': dynamic_pressure,
    'D': drag
})

pd.set_option('display.max_rows', None)

df_filtered = df[df.index % 10000 == 0].reset_index(drop=True)

max_values = df[['h', 'v', 'mach', 'q', 'D']].max().to_frame().T
#max_values['time'] = 'max'

print(df_filtered)
print(max_values)

plt.figure(figsize=(18, 10))

plt.subplot(3, 2, (1, 5))
plt.plot(distance, altitude, marker='.', linestyle='-')
plt.title('Trajectory')
plt.xlabel('x distance (m)')
plt.ylabel('Altitude (m)')
plt.axis('equal')  
plt.grid(True)     

plt.subplot(3, 2, 2)  
plt.plot(time, altitude, marker='.', linestyle='-')
plt.title('Altitude vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')

plt.subplot(3, 2, 4)  
plt.plot(time, dynamic_pressure, marker='.', linestyle='-')
plt.title('Dynamic Pressure vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Dynamic Pressure (Pa)')

plt.subplot(3, 2, 6)  
plt.plot(time, drag, marker='.', linestyle='-')
plt.title('Drag vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Drag (N)')

plt.tight_layout()
plt.show()
