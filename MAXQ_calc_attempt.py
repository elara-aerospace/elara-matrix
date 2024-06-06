"""
Attempt to calculate altitude, velocity and dynamic pressure of a rocket in an interative system

This code uses the thrust force T, air resistance Fd and gravity to calculate the trajectory over a given time of a rocket.
For the Air resistance I used the atmosphereic model from NASA https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/earth-atmosphere-equation-metric/
The used drac coeefcient is that of a cone. I dont know ours yet nor do i know how i evolves over density and velocity changes.
This model has no calculation for lift forces
The coordinate system is a x-y-z System but we only look at the x-z Plane. x horizontal, z vertical
mx_k is a matrix with k1 and k1 as factors on it diagonal. they are trig-functions. at the moment you can plug in an angle of 0° - 90°
0° would mean the rocket starts vertical, and 90° would mean a horizontal start, wich obv. makes no sense.
in the mx_k def is teh facto f which will be a function in the future to make the rocket start vertically and then let it change its angle over time
the base formual is a*m = T - m*g - Fd which is a differential equation. This skript is an attempt to bypass the need of solving this equation
m is a constat function of time. so we have a = T/m - g - Fd/m . a*dt with dt beeing really small we get a velocity, wich we add on our initial velocity
so we have a1 wich leads to dv1 so we have v0 + dv1 = v1. v1*dt = dh1 and h0 + dh1 = h1. a1, v1 and h1 are the new values for the next calulation, wich will-
-result in a2. a2 will result in v2 and h2 and the cycle repeats. and the smaller dt is, the more precice the calculation gets.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   Formulas
# ------------------------------------------------------------------------------------------------------------------------------------------------


def area(radius):
    """Calculate the cross-section area of the rocket."""
    return math.pi * radius**2


def mass_over_time(mass, fuel_consumption, time):
    """Calculate the mass over time, considering fuel consumption."""
    return mass - fuel_consumption * time


def density(height):
    """Calculate the air density based on the height."""
    if height < 0:
        raise ValueError("Error: negative height value")

    elif 0 <= height < 11000:  # Troposphere
        temp = 15.04 - 0.00649 * height + 273.1
        pressure = 101.29 * (temp / 288.08) ** 5.256

    elif 11000 <= height < 25000:  # Lower Stratosphere
        temp = -56.46 + 273.1
        pressure = 22.65 * math.exp(1.73 - 0.000157 * height)

    elif height >= 25000:  # Upper Stratosphere
        temp = -131.21 + 0.00299 * height + 273.1
        pressure = 2.488 * (temp / 216.6) ** (-11.388)

    else:
        raise ValueError("Error: height value out of range")

    density_value = pressure / (0.2869 * temp)  # density in kg/m^3
    return density_value


def dynamic_pressure(density_value, velocity):
    """Calculate the dynamic pressure in Pa = N/m^2."""
    return 0.5 * density_value * velocity**2


def air_resistance(dynamic_pressure, drag_coefficient, area):
    """Calculate the air resistance. F = 1/2 * Cd * A * d(density) * v^2. Because 1/2 * d * v^2 is q, we can substitute it."""
    return dynamic_pressure * area * drag_coefficient


def thrust(time):
    """Calculate the thrust over time. Burnout after 76s after Launch."""
    if time <= 76:
        return 28000
    else:
        return 0


def gravity(height):
    """Calculate the gravitational factor/the big number is a factor. Mass of earth and G. The constant is chosen so that at h_0, g(h) = g_0."""
    return 398184378.21 / (6371 + (height / 1000)) ** 2


def rotation_matrix(time):
    """Calculate the rotation matrix for the rocket's angle."""
    angle = 0 * time + 0.01745329 * 1  # Angle in radians
    k1 = np.cos(math.pi / 2 - angle)
    k2 = np.sin(math.pi / 2 - angle)
    rotation_matrix = np.array([[k1, 0], [0, k2]])
    return rotation_matrix


def sign(value):
    """Calculate the sign of a value."""
    if value == 0:
        return 0
    elif value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        raise ValueError("Error: issue with sign function")

# ------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   Values
# ------------------------------------------------------------------------------------------------------------------------------------------------

DRAG_COEFFICIENT = 0.5  # no unit
RADIUS = 0.225  # m
INITIAL_TIME = 0  # s
INITIAL_VELOCITY = 0  # m/s
INITIAL_HEIGHT = 0
INITIAL_DISTANCE = 0  # m
BURNOUT_TIME = 76  # s
THRUST_CONSTANT = 28000  # N
INITIAL_MASS = 1000  # kg
FUEL_CONSUMPTION = 9.32454  # kg/s
GRAVITATIONAL_ACCELERATION = 9.81  # m/s^2
SPEED_OF_SOUND = 331  # m/s
PI = round(np.pi, 6)

# ------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                  Calculation
# ------------------------------------------------------------------------------------------------------------------------------------------------

altitude = []
distance = []
dynamic_pressure_values = []
drag_values = []
time_values = []
velocity_values = []
mach_numbers = []
thrust_values = []
gravitational_potential = []

time = INITIAL_TIME
position_vector = np.array([[INITIAL_DISTANCE], [INITIAL_HEIGHT]])
vx = INITIAL_VELOCITY
vz = INITIAL_VELOCITY
velocity_vector = np.array([[vx], [vz]])
dt = 0.001
time_step_matrix = np.array([[dt, 0], [0, dt]])
dv = 0
sign_matrix = np.array([[sign(dv), 0], [0, sign(dv)]])

max_time = 750

while time < max_time:

    # Base values
    x_position = position_vector[0, 0]
    height = position_vector[1, 0]
    velocity = (velocity_vector[0, 0] ** 2 + velocity_vector[1, 0] ** 2) ** 0.5

    # Check
    if time > 0.01 * max_time:
        if height <= 0:
            break

    mach_number = velocity / SPEED_OF_SOUND
    current_mass = mass_over_time(INITIAL_MASS, FUEL_CONSUMPTION, time)
    current_thrust = thrust(time)
    current_dynamic_pressure = dynamic_pressure(density(height), velocity)
    current_drag = air_resistance(current_dynamic_pressure, DRAG_COEFFICIENT, area(RADIUS))

    # Accelerations
    current_gravity = gravity(height)
    acceleration_thrust = current_thrust / current_mass
    acceleration_drag = current_drag / current_mass

    # Lists for evaluation
    time_values.append(time)
    thrust_values.append(current_thrust)
    altitude.append(round(height, 5))
    distance.append(round(x_position, 5))
    velocity_values.append(round(velocity, 5))
    mach_numbers.append(round(mach_number, 3))
    dynamic_pressure_values.append(round(current_dynamic_pressure, 5))
    drag_values.append(round(current_drag, 5))
    gravitational_potential.append(round(current_gravity, 5))

    # Vectors
    acceleration_thrust_vector = np.array([[acceleration_thrust], [acceleration_thrust]])
    gravity_vector = np.array([[0], [current_gravity]])
    acceleration_drag_vector = np.array([[acceleration_drag], [acceleration_drag]])

    velocity_change_vector = time_step_matrix @ (rotation_matrix(time) @ (acceleration_thrust_vector - sign_matrix @ acceleration_drag_vector) - gravity_vector)

    dv = velocity_change_vector[1, 0]

    velocity_vector = velocity_vector + velocity_change_vector

    position_change_vector = time_step_matrix @ velocity_vector

    position_vector = position_vector + position_change_vector

    # Time iteration
    time = time + dt

# ------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   Evaluation
# ------------------------------------------------------------------------------------------------------------------------------------------------

# Generate a table showing all calculated values at whole seconds
data = {
    "time": time_values,
    "thrust": thrust_values,
    "height": altitude,
    "velocity": velocity_values,
    "mach": mach_numbers,
    "gravity": gravitational_potential,
    "dynamic_pressure": dynamic_pressure_values,
    "drag": drag_values,
}

df = pd.DataFrame(data)

pd.set_option("display.max_rows", None)

df_filtered = df[df.index % 10000 == 0].reset_index(drop=True)

max_values = df[["height", "velocity", "mach", "dynamic_pressure", "drag"]].max().to_frame().T
# max_values['time'] = 'max'

print(df_filtered)
print(max_values)

plt.figure(figsize=(18, 10))

plt.subplot(3, 2, (1, 5))
plt.plot(distance, altitude, marker=".", linestyle="-")
plt.title("Trajectory")
plt.xlabel("x distance (m)")
plt.ylabel("Altitude (m)")
plt.axis("equal")
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(time_values, altitude, marker=".", linestyle="-")
plt.title("Altitude vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")

plt.subplot(3, 2, 4)
plt.plot(time_values, dynamic_pressure_values, marker=".", linestyle="-")
plt.title("Dynamic Pressure vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Dynamic Pressure (Pa)")

plt.subplot(3, 2, 6)
plt.plot(time_values, drag_values, marker=".", linestyle="-")
plt.title("Drag vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Drag (N)")

plt.tight_layout()
plt.show()
