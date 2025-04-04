import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from motion_models.quadrotor import QuadrotorModel

# Initialize model with physical parameters from paper
dt = 0.01
quad = QuadrotorModel(
    dt=dt,
    mc=1.2,        # Mass (kg)
    lo=0.25,       # Arm length (m)
    Ixx=0.1,       # X inertia
    Iyy=0.1,       # Y inertia
    Izz=0.2        # Z inertia
)

# Initial state [x,y,z, x_dot,y_dot,z_dot, phi,theta,psi, phi_dot,theta_dot,psi_dot]
initial_state = np.zeros(12)
initial_state[2] = 1.0  # Start at 1m altitude

# Simulation parameters
sim_time = 10  # seconds
num_steps = int(sim_time / dt)
time = np.arange(0, sim_time + dt, dt)

# Control sequence definition -------------------------------------------------
controls = np.zeros((num_steps, 4))
hover_thrust = quad.mc * quad.g  # Calculated from physics

# Phase 1: Hover stabilization (0-2s)
controls[0:int(2/dt), 0] = hover_thrust

# Phase 2: Vertical takeoff (2-4s)
takeoff_steps = int(2/dt)
controls[takeoff_steps:takeoff_steps+int(2/dt), 0] = 1.5 * hover_thrust

# Phase 3: Forward flight (4-6s)
forward_steps = int(2/dt)
idx = takeoff_steps*2
controls[idx:idx+forward_steps, 0] = hover_thrust
controls[idx:idx+forward_steps, 2] = 0.15  # Positive pitch for forward motion

# Phase 4: Coordinated turn (6-8s)
turn_steps = int(2/dt)
controls[idx+forward_steps:idx+forward_steps+turn_steps, 0] = hover_thrust
controls[idx+forward_steps:idx+forward_steps+turn_steps, 1] = 0.1  # Roll right
controls[idx+forward_steps:idx+forward_steps+turn_steps, 3] = 0.05 # Yaw right

# Phase 5: Controlled descent (8-10s)
controls[idx+forward_steps+turn_steps:, 0] = 0.8 * hover_thrust

# Simulation -------------------------------------------------------------------
states = np.zeros((num_steps + 1, 12))
states[0] = initial_state

for i in range(num_steps):
    states[i+1] = quad.forward_dynamics(states[i], controls[i])

# Plotting ---------------------------------------------------------------------

# Plot 1: 3D Trajectory
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(states[:,0], states[:,1], states[:,2], 'b-', lw=2)
ax1.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]', title='3D Flight Path')
plt.show()

# Plot 2: Position States
fig2 = plt.figure()
plt.plot(time, states[:,0], 'r-', label='X')
plt.plot(time, states[:,1], 'g-', label='Y')
plt.plot(time, states[:,2], 'b-', label='Z')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Linear Position')
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Euler Angles
fig3 = plt.figure()
plt.plot(time, np.rad2deg(states[:,6]), 'r-', label='Roll (φ)')
plt.plot(time, np.rad2deg(states[:,7]), 'g-', label='Pitch (θ)')
plt.plot(time, np.rad2deg(states[:,8]), 'b-', label='Yaw (ψ)')
plt.xlabel('Time [s]')
plt.ylabel('Degrees')
plt.title('Attitude Angles')
plt.legend()
plt.grid(True)
plt.show()

# Plot 4: Control Inputs
fig4 = plt.figure()
plt.plot(time[:-1], controls[:,0], 'm-', label='Thrust (u1)')
plt.plot(time[:-1], controls[:,1], 'c-', label='Roll Torque (u2)')
plt.plot(time[:-1], controls[:,2], 'y-', label='Pitch Torque (u3)')
plt.plot(time[:-1], controls[:,3], 'k-', label='Yaw Torque (u4)')
plt.xlabel('Time [s]')
plt.ylabel('Control Input')
plt.title('Control Commands')
plt.legend()
plt.grid(True)
plt.show()
