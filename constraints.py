import numpy as np

class CircleConstraintForCar:
	def __init__(self, center, r, system):
		self.center = center
		self.r = r
		self.system = system
	def evaluate_constraint(self, x):	
		#evolve the system for one to evaluate constraint
		x_next = self.system.transition(x, np.zeros(self.system.control_size))
		length = (x_next[0] - self.center[0])**2 + (x_next[1] - self.center[1])**2
		#print(x_next, self.r**2 - length)
		return self.r**2 - length
	def evaluate_constraint_J(self, x):
		#evolve the system for one to evaluate constraint
		x_next = self.system.transition(x, np.zeros(self.system.control_size))
		result = np.zeros(x.shape)
		result[0] = -2*(x_next[0] - self.center[0])
		result[1] = -2*(x_next[1] - self.center[1])
		result[2] = -2*(x_next[0] - self.center[0]) * self.system.dt * x[3] * np.cos(x[2]) + 2*(x_next[1] - self.center[1]) * self.system.dt * x[3] * np.sin(x[2])
		result[3] = -2*(x_next[1] - self.center[1]) * self.system.dt * np.cos(x[2]) -2*(x_next[0] - self.center[0]) * self.system.dt * np.sin(x[2])
		return result

class CircleConstraintForDoubleIntegrator:
	def __init__(self, center, r, system):
		self.center = center
		self.r = r
		self.system = system
	def evaluate_constraint(self, x):	
		#evolve the system for one to evaluate constraint
		x_next = self.system.transition(x, np.zeros(self.system.control_size))
		length = (x_next[0] - self.center[0])**2 + (x_next[1] - self.center[1])**2
		#print(x_next, self.r**2 - length)
		return self.r**2 - length
	def evaluate_constraint_J(self, x):
		#evolve the system for one to evaluate constraint
		x_next = self.system.transition(x, np.zeros(self.system.control_size))
		result = np.zeros(x.shape)
		result[0] = -2*(x_next[0] - self.center[0])
		result[1] = -2*(x_next[1] - self.center[1])
		result[2] = -2*(x_next[0] - self.center[0]) * self.system.dt
		result[3] = -2*(x_next[1] - self.center[1]) * self.system.dt
		return result

class SphereConstraintForQuadrotor:
	def __init__(self, center, r, system):
		self.center = center
		self.r = r
		self.system = system

	def evaluate_constraint(self, x):
		x_next = self.system.transition(x, np.zeros(self.system.control_size))
		length = (x_next[0] - self.center[0])**2 + (x_next[1] - self.center[1])**2 + (x_next[2] - self.center[2])**2
		return self.r**2 - length

	def evaluate_constraint_J(self, x):
		x_next = self.system.transition(x, np.zeros(self.system.control_size))
		result = np.zeros(x.shape)
		result[0] = -2*(x_next[0] - self.center[0])
		result[1] = -2*(x_next[1] - self.center[1])
		result[2] = -2*(x_next[2] - self.center[2])
		result[3] = -2*(x_next[0] - self.center[0]) * self.system.dt
		result[4] = -2*(x_next[1] - self.center[1]) * self.system.dt
		result[5] = -2*(x_next[2] - self.center[2]) * self.system.dt
		return result
