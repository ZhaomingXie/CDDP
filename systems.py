import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DynamicalSystem:
	def __init__(self, state_size, control_size):
		self.state_size = state_size
		self.control_size = control_size
	def set_cost(self, Q, R):
		# one step cost = x.T * Q * x + u.T * R * u
		self.Q = Q
		self.R = R
	def set_final_cost(self, Q_f):
		self.Q_f = Q_f
	def calculate_cost(self, x, u):
		return 0.5*((x-self.goal).T.dot(self.Q).dot(x-self.goal) + u.T.dot(self.R).dot(u))
	def calculate_final_cost(self, x):
		return 0.5*(x-self.goal).T.dot(self.Q_f).dot(x-self.goal)
	def set_goal(self, x_goal):
		self.goal = x_goal

class DoubleIntegrator(DynamicalSystem):
	def __init__(self):
		super().__init__(4, 2)
		self.dt = 0.05
		self.control_bound = np.ones(2) * 100
		self.goal = np.zeros(4)
	def transition(self, x, u):
		result = np.zeros(4)
		result[0:2] = x[0:2] + self.dt * x[2:4]
		result[2:4] = x[2:4] + self.dt * u
		return result
	def transition_J(self, x, u):
		#return matrix A, B, so that x = Ax + Bu
		A = np.zeros((self.state_size, self.state_size))
		B = np.zeros((self.state_size, self.control_size))
		A[0:self.state_size, 0:self.state_size] = np.identity(self.state_size)
		A[0, 2] = self.dt
		A[1, 3] = self.dt
		B[2, 0] = self.dt
		B[3, 1] = self.dt
		return A, B
	def draw_trajectories(self, x_trajectories):
		ax = plt.subplot(111)
		circle1 = plt.Circle((1, 1), 0.5, color=(0, 0.8, 0.8))
		circle2 = plt.Circle((1.5, 2.2), 0.5, color=(0, 0.8, 0.8))
		ax.add_artist(circle1)
		ax.add_artist(circle2)
		plt.scatter(x_trajectories[0, 0::5], x_trajectories[1, 0::5], 4,color='r')
		ax.set_aspect("equal")
		ax.set_xlim(0, 3)
		ax.set_ylim(0, 3)
		plt.show()
	def draw_u_trajectories(self, u_trajectories):
		x = plt.subplot(111)
		plt.scatter(u_trajectories[0, 0::5], u_trajectories[1, 0::5], 4,color='r')
		plt.show()

class Car(DynamicalSystem):
	def __init__(self):
		super().__init__(4, 2)
		self.dt = 0.05
		self.control_bound = np.array([np.pi/2, 10])
		self.goal = np.zeros(4)
	def transition(self, x, u):
		x_next = np.zeros(4)
		x_next[0] = x[0] + self.dt * x[3] * np.sin(x[2])
		x_next[1] = x[1] + self.dt * x[3] * np.cos(x[2])
		x_next[2] = x[2] + self.dt * u[1] * x[3]
		x_next[3] = x[3] + self.dt * u[0]
		return x_next
	def transition_J(self, x, u):
		A = np.identity(4)
		B = np.zeros((4, 2))
		A[0, 3] = np.sin(x[2]) * self.dt
		A[0, 2] = x[3] * np.cos(x[2]) * self.dt
		A[1, 3] = np.cos(x[2]) * self.dt
		A[1, 2] = -x[3] * np.sin(x[2]) * self.dt
		A[2, 3] = u[1] * self.dt
		B[2, 1] = x[3] * self.dt
		B[3, 0] = self.dt
		return A, B
	def draw_trajectories(self, x_trajectories):
		ax = plt.subplot(111)
		circle1 = plt.Circle((1, 1), 0.5, color=(0, 0.8, 0.8))
		circle2 = plt.Circle((2, 2), 1, color=(0, 0.8, 0.8))
		ax.add_artist(circle1)
		ax.add_artist(circle2)
		for i in range(0, x_trajectories.shape[1]-1, 5):
			circle_car = plt.Circle((x_trajectories[0, i], x_trajectories[1, i]), 0.1, facecolor='none')
			ax.add_patch(circle_car)
			ax.arrow(x_trajectories[0, i], x_trajectories[1, i], 0.1*np.sin(x_trajectories[2, i]), 0.1 * np.cos(x_trajectories[2, i]), head_width=0.05, head_length=0.1, fc='k', ec='k')
		ax.set_aspect("equal")
		ax.set_xlim(-1, 4)
		ax.set_ylim(-1, 4)
		plt.show()

class Quadrotor(DynamicalSystem):
	def __init__(self):
		super().__init__(12, 4)
		self.dt = 0.02
		self.control_bound = np.array([10, 10, 10, 10])
		self.goal = np.zeros(12)

	def transition(self, x, u):
		forces = np.array([0, 0, u[0]+u[1]+u[2]+u[3]])
		torques = np.array([u[0]-u[2], u[1] - u[3], u[0] - u[1] + u[2] - u[3]])
		rotation_matrix = self.get_rotation_matrix(x)
		J_omega = self.get_J_omega(x)
		g = np.array([0, 0, -10])
		x_next = np.zeros(12)
		x_next[0:3] = x[0:3] + self.dt * x[6:9]
		x_next[6:9] = x[6:9] + self.dt * (g + rotation_matrix.dot(forces) - 0 * x[6:9])
		x_next[3:6] = x[3:6] + self.dt * J_omega.dot(x[9:12])
		x_next[9:12] = x[9:12] + self.dt * torques
		return x_next

	def transition_J(self, x, u):
		A = np.zeros((self.state_size, self.state_size))
		B = np.zeros((self.state_size, self.control_size))
		u_sum = u[0]+u[1]+u[2]+u[3]
		rotation_matrix = self.get_rotation_matrix(x)
		A[0:self.state_size, 0:self.state_size] = np.identity(self.state_size)
		A[0, 6] = 1 * self.dt
		A[1, 7] = 1 * self.dt
		A[2, 8] = 1 * self.dt
		
		# A[6, 3] = u_sum * self.dt * (-np.cos(x[5]) * np.sin(x[4]) * np.sin(x[3]) + np.sin(x[5]) * np.cos(x[3]))
		# A[6, 4] = u_sum * self.dt * (np.cos(x[5]) * np.cos(x[4]) * np.cos(x[3]))
		# A[6, 5] = u_sum * self.dt * (-np.sin(x[5]) * np.sin(x[4]) * np.cos(x[3]) + np.cos(x[5]) * np.sin(x[3]))
		# A[7, 3] = u_sum * self.dt * (-np.sin(x[5]) * np.sin(x[4]) * np.sin(x[3]) - np.cos(x[5]) * np.cos(x[3]))
		# A[7, 4] = u_sum * self.dt * (np.sin(x[5]) * np.cos(x[4]) * np.cos(x[3]))
		# A[7, 5] = u_sum * self.dt * (np.cos(x[5]) * np.sin(x[4]) * np.cos(x[3]) + np.sin(x[5]) * np.sin(x[3]))
		# A[8, 3] = u_sum * self.dt * (-np.cos(x[4]) * np.sin(x[3]))
		# A[8, 4] = u_sum * self.dt * (-np.sin(x[4]) * np.cos(x[3]))
		
		A[6, 3] += u_sum * self.dt * (np.sin(x[4]) * np.cos(x[3]))
		A[6, 4] += u_sum * self.dt * (np.cos(x[4]) * np.sin(x[3]))
		A[7, 4] += u_sum * self.dt * (-np.cos(x[5]) * np.cos(x[4]))
		A[7, 5] += u_sum * self.dt * (np.sin(x[5]) * np.sin(x[4]))
		A[8, 4] += u_sum * self.dt * (-np.sin(x[4]))

		# A[6, 6] -= self.dt
		# A[7, 7] -= self.dt
		# A[8, 8] -= self.dt

		A[3, 3] += (x[10] * np.cos(x[3]) * np.tan(x[4]) - x[11] * np.sin(x[3]) * np.tan(x[4])) * self.dt
		A[3, 4] += self.dt * 1.0 / (np.cos(x[4])**2) * (x[10] * np.sin(x[3]) + x[11] * np.cos(x[3]))
		A[3, 9] += self.dt
		A[3, 10] += np.sin(x[3]) * np.tan(x[4]) * self.dt
		A[3, 11] += np.cos(x[3]) * np.tan(x[4]) * self.dt
		A[4, 3] += self.dt * (-np.sin(x[3]) * x[10] - np.cos(x[3]) * x[11])
		A[4, 10] += self.dt * np.cos(x[3])
		A[4, 11] += -self.dt * np.sin(x[3])
		A[5, 3] += self.dt * (np.cos(x[3]) / np.cos(x[4]) * x[10] - np.sin(x[3]) / np.cos(x[4]) * x[11])
		A[5, 4] += self.dt * np.sin(x[4]) / (np.cos(x[4])**2) * (np.sin(x[3]) * x[10] + np.cos(x[3]) * x[11])
		A[5, 10] += self.dt * np.sin(x[3]) / np.cos(x[4])
		A[5, 11] += self.dt * np.cos(x[3]) / np.cos(x[4])

		torque_matrix = np.zeros((3, 4))
		torque_matrix[0, 0] = 1
		torque_matrix[0, 2] = -1
		torque_matrix[1, 1] = 1
		torque_matrix[1, 3] = -1
		torque_matrix[2, 0] = 1
		torque_matrix[2, 1] = -1
		torque_matrix[2, 2] = 1
		torque_matrix[2, 3] = -1

		force_matrix = np.zeros((3, 4))
		force_matrix[2, :] = 1
		
		B[9:12, :] = torque_matrix * self.dt
		B[6:9, :] = self.dt * rotation_matrix.dot(force_matrix)
		return A, B

	def get_rotation_matrix(self, x):
		R = np.zeros((3, 3))
		# R[0, 0] = np.cos(x[5]) * np.cos(x[4])
		# R[0, 1] = np.cos(x[5]) * np.sin(x[4]) * np.sin(x[3]) - np.sin(x[5]) * np.cos(x[3])
		# R[0, 2] = np.cos(x[5]) * np.sin(x[4]) * np.cos(x[3]) + np.sin(x[5]) * np.sin(x[3])
		# R[1, 0] = np.sin(x[5]) * np.cos(x[4])
		# R[1, 1] = np.sin(x[5]) * np.sin(x[4]) * np.sin(x[3]) + np.cos(x[5]) * np.cos(x[3])
		# R[1, 2] = np.sin(x[5]) * np.sin(x[4]) * np.cos(x[3]) - np.cos(x[5]) * np.sin(x[3])
		# R[2, 0] = -np.sin(x[4])
		# R[2, 1] = np.cos(x[4]) * np.sin(x[3])
		# R[2, 2] = np.cos(x[4]) * np.cos(x[3])
		
		R[0, 0] = np.cos(x[3]) * np.cos(x[5]) - np.cos(x[4]) * np.sin(x[3]) * np.sin(x[5])
		R[0, 1] = -np.cos(x[3]) * np.sin(x[3]) - np.cos(x[3]) * np.cos(x[4]) * np.sin(x[5])
		R[0, 2] = np.sin(x[4]) * np.sin(x[3])
		R[1, 0] = np.cos(x[4]) * np.cos(x[5]) * np.sin(x[3]) 
		R[1, 1] = np.cos(x[3]) * np.cos(x[4]) * np.cos(x[5]) - np.sin(x[3]) * np.sin(x[5])
		R[1, 2] = -np.cos(x[5]) * np.sin(x[4])
		R[2, 0] = np.sin(x[3]) * np.sin(x[4])
		R[2, 1] = np.cos(x[3]) * np.sin(x[4])
		R[2, 2] = np.cos(x[4])
		return R

	def get_J_omega(self, x):
		J_omega = np.zeros((3, 3))
		J_omega[0, 0] = 1
		J_omega[0, 1] = np.sin(x[3]) * np.tan(x[4])
		J_omega[0, 2] = np.cos(x[3]) * np.tan(x[4])
		J_omega[1, 1] = np.cos(x[3])
		J_omega[1, 2] = -np.sin(x[3])
		J_omega[2, 1] = np.sin(x[3]) / np.cos(x[4])
		J_omega[2, 2] = np.cos(x[3]) / np.cos(x[4])
		return J_omega

if __name__ == '__main__':
	system = Quadrotor()
	initial_state = np.array([-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	initial_u = np.array([2.5, 2.5, 2.5, 2.5])
	perturb_u = np.array([0.01, 0, 0, 0])
	perturb_x = np.zeros(12)
	perturb_x[4] += 0.01
	A, B = system.transition_J(initial_state, initial_u)
	next_state = system.transition(initial_state+perturb_x, initial_u+perturb_u)
	linearized_next_state = system.transition(initial_state, initial_u) + A.dot(perturb_x)+ B.dot(perturb_u)
	print(next_state-linearized_next_state)