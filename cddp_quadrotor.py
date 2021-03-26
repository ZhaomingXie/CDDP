import numpy as np
from numpy.linalg import inv
import time
import osqp
import scipy.sparse as sparse

from systems import DoubleIntegrator, Car
from constraints import CircleConstraintForDoubleIntegrator, CircleConstraintForCar

class CDDP:
	def __init__(self, system, initial_state, horizon=300):
		self.system = system
		self.horizon = horizon
		self.x_trajectories = np.zeros((self.system.state_size, self.horizon + 1))
		self.u_trajectories = np.ones((self.system.control_size, self.horizon)) * 2.5
		# self.u_trajectories[0, :] = 2.5
		# self.u_trajectories[1, :] = 2.0
		# self.u_trajectories[2, :] = 2.5
		# self.u_trajectories[3, :] = 2.5
		self.initial_state = np.copy(initial_state)
		self.constraints = []
		self.best_J = 100000000000
		self.Q_UX = np.zeros((self.system.control_size, self.system.state_size, self.horizon))
		self.Q_UU = np.zeros((self.system.control_size, self.system.control_size, self.horizon))
		self.Q_U = np.zeros((self.system.control_size, self.horizon))
		self.reg_factor = 0.1
		self.reg_factor_u = 0.001
		self.active_set_tol = 0.01

	def set_initial_trajectories(self, x_trajectories, u_trajectories):
		self.x_trajectories = np.copy(x_trajectories)
		self.u_trajectories = np.copy(u_trajectories)

	def add_constraint(self, constraint):
		self.constraints.append(constraint)
	
	def forward_pass(self):
		x = np.copy(self.initial_state)
		feasible = False
		trust_region_scale = 1
		while not feasible:
			feasible = True
			current_J = 0
			x_new_trajectories = np.zeros((self.system.state_size, self.horizon + 1))
			u_new_trajectories = np.zeros((self.system.control_size, self.horizon))
			x = np.copy(self.initial_state)
			x_new_trajectories[:, 0] = np.copy(x)
			for i in range(self.horizon):
				delta_x = x - self.x_trajectories[:, i]
				x_new_trajectories[:, i] = np.copy(x)
				Q_ux = self.Q_UX[:, :, i]
				Q_u = self.Q_U[:, i]
				P = sparse.csr_matrix(self.Q_UU[:, : , i])
				q = (Q_ux.dot(delta_x) + Q_u)
				'''lb = -self.system.control_bound - self.u_trajectories[:, i]
				ub = self.system.control_bound - self.u_trajectories[:, i]
				lb *= trust_region_scale
				ub *= trust_region_scale'''

				#constraint_A = sparse.csr_matrix(np.identity(self.system.control_size))

				#initialize contraint matrix and bound
				constraint_A = np.zeros((self.system.control_size + len(self.constraints), self.system.control_size))
				lb = np.zeros(self.system.control_size + len(self.constraints))
				ub = np.zeros(self.system.control_size + len(self.constraints))

				#control limit contraint
				constraint_A[0:self.system.control_size, 0:self.system.control_size] = np.identity(self.system.control_size)
				lb[0:self.system.control_size] = 0 - self.u_trajectories[:, i]
				ub[0:self.system.control_size] = self.system.control_bound - self.u_trajectories[:, i]
				lb *= trust_region_scale
				ub *= trust_region_scale

				#formulate linearized state constraints
				f_x, f_u = self.system.transition_J(x, self.u_trajectories[:, i])				
				constraint_index = self.system.control_size
				for constraint in self.constraints:
					if i <= self.horizon - 2:#current action might cause state constraint violation
						x_temp = self.system.transition(x, self.u_trajectories[:, i])
						D = constraint.evaluate_constraint(x_temp)
						#print("constraint eval", D, i, x)
						C = constraint.evaluate_constraint_J(x_temp)
						#print(C.shape, f_u.shape)
						C = C.dot(f_u)
						constraint_A[constraint_index, :] = np.copy(C)
						lb[constraint_index] = -np.inf #no lower bound
						ub[constraint_index] = -D
					constraint_index += 1

				constraint_A = sparse.csr_matrix(constraint_A)
				prob = osqp.OSQP()
				prob.setup(P, q, constraint_A, lb, ub, alpha=1.0, verbose=False)
				res = prob.solve()
				if res.info.status != 'solved':
					feasible = False
					#print("infeasible, reduce trust region")
					trust_region_scale *= 0.5
					break
				delta_u = res.x[0:self.system.control_size]
				u = delta_u + self.u_trajectories[:, i]
				u_new_trajectories[:, i] = np.copy(u)
				current_J += self.system.calculate_cost(x, u)
				x = self.system.transition(x, u)
			x_new_trajectories[:, self.horizon] = np.copy(x)
			current_J += self.system.calculate_final_cost(x)
			if current_J > self.best_J:
				feasible = False
				trust_region_scale *= 0.5
			else:
				self.best_J = current_J
			if feasible == True:
				self.x_trajectories = np.copy(x_new_trajectories)
				self.u_trajectories = np.copy(u_new_trajectories)
				print("total cost", current_J)
				print("end traj", x)
				#self.system.draw_trajectories(self.x_trajectories)
				#self.system.draw_u_trajectories(self.u_trajectories)

	def backward_pass(self):
		A = np.copy(self.system.Q_f)
		b = self.system.Q_f.dot(self.x_trajectories[:, self.horizon] - self.system.goal)
		for i in range(self.horizon - 1, -1, -1):
			u = self.u_trajectories[:, i]
			x = self.x_trajectories[:, i]
			l_xt = self.system.Q.dot(x - self.system.goal)
			l_ut = self.system.R.dot(u)
			l_uxt = np.zeros((self.system.control_size, self.system.state_size))
			l_xxt = np.copy(self.system.Q)
			l_uut = np.copy(self.system.R)
			f_x, f_u = self.system.transition_J(x, u)
			Q_x = l_xt + (f_x.T).dot(b)
			Q_u = l_ut + (f_u.T).dot(b)
			Q_xx = l_xxt + f_x.T.dot(A + self.reg_factor * np.identity(self.system.state_size)).dot(f_x)
			Q_ux = l_uxt + f_u.T.dot(A + self.reg_factor * np.identity(self.system.state_size)).dot(f_x)
			Q_uu = l_uut + f_u.T.dot(A + self.reg_factor * np.identity(self.system.state_size)).dot(f_u) + self.reg_factor_u * np.identity(self.system.control_size)

			print(f_u)
			
			#identify active constraint
			C = np.empty((self.system.control_size + len(self.constraints), self.system.control_size))
			D = np.empty((self.system.control_size + len(self.constraints), self.system.state_size))
			index = 0
			constraint_index = np.zeros((2 * self.system.control_size + len(self.constraints) * self.system.state_size, self.horizon))
			for j in range(self.system.control_size):
				if u[j] >= self.system.control_bound[j] - self.active_set_tol:
					e = np.zeros(self.system.control_size)
					e[j] = 1
					C[index, :] = e
					D[index, :] = np.zeros(self.system.state_size)
					index += 1
					constraint_index[j, i] = 1
				elif u[j] <= 0 + self.active_set_tol:
					e = np.zeros(self.system.control_size)
					e[j] = -1
					C[index, :] = e
					D[index, :] = np.zeros(self.system.state_size)
					index += 1
					constraint_index[j + self.system.control_size, i] = 1
			if i <= self.horizon - 2: #state constraint can be violated
				for j in range(len(self.constraints)):
					D_constraint = self.constraints[j].evaluate_constraint(self.x_trajectories[:, i+1])
					#print("constraint", D_constraint, i)
					if abs(D_constraint) <= self.active_set_tol:
						C_constraint = self.constraints[j].evaluate_constraint_J(self.x_trajectories[:, i+1])
						C[index, :] = C_constraint.dot(f_u)
						#print(C_constraint.dot(f_u))
						D[index, :] = -C_constraint.dot(f_x)
						index = index + 1
						constraint_index[2 * self.system.control_size + j, i] = 1
			
			if index == 0: #no constraint active
				K = -inv(Q_uu).dot(Q_ux)
				k = -inv(Q_uu).dot(Q_u)
			else:
				C = C[0:index, :]
				D = D[0:index, :]
				lambda_temp = C.dot(inv(Q_uu)).dot(C.T)
				lambda_temp = -inv(lambda_temp).dot(C).dot(inv(Q_uu)).dot(Q_u)

				#remove active constraint with lambda < 0
				index = 0
				delete_index = []
				#control constraint
				for j in range(self.system.control_size):
					if constraint_index[j, i] == 1:
						if lambda_temp[index] < 0:
							constraint_index[j, i] = 0
							C[index, :] = np.zeros(self.system.control_size)
							delete_index.append(index)
						index = index + 1
					elif constraint_index[j + self.system.control_size, i] == 1:
						if lambda_temp[index] < 0:
							constraint_index[j + self.system.control_size, i] = 0
							C[index, :] = np.zeros(self.system.control_size)
							delete_index.append(index)
						index = index + 1
				#state constrait
				for j in range(len(self.constraints)):
					if constraint_index[j + 2 * self.system.control_size, i] == 1:
						if lambda_temp[index] < 0:
							constraint_index[j + 2 * self.system.control_size, i] = 0
							C[index, :] = np.zeros(self.system.control_size)
							delete_index.append(index)
						index += 1

				if len(delete_index) < C.shape[0]:
					C = np.delete(C, delete_index, axis=0)
					D = np.delete(D, delete_index, axis=0)
					C_star = inv(C.dot(inv(Q_uu)).dot(C.T)).dot(C).dot(inv(Q_uu))
					H_star = inv(Q_uu).dot(np.identity(self.system.control_size) - C.T.dot(C_star))
					k = -H_star.dot(Q_u)
					K = -H_star.dot(Q_ux) + C_star.T.dot(D)
				else:
					K = -inv(Q_uu).dot(Q_ux)
					k = -inv(Q_uu).dot(Q_u)
			A = Q_xx + K.T.dot(Q_uu).dot(K) + Q_ux.T.dot(K) + K.T.dot(Q_ux)
			b = Q_x + Q_ux.T.dot(k) + K.T.dot(Q_uu).dot(k) + K.T.dot(Q_u)
			self.Q_UX[:, :, i] = Q_ux
			self.Q_UU[:, :, i] = Q_uu
			self.Q_U[:, i] = Q_u

if __name__ == '__main__':
	system = Car()
	system.set_cost(np.zeros((4, 4)), 0.05*np.identity(2))
	Q_f = np.identity(4)
	Q_f[0, 0] = 50
	Q_f[1, 1] = 50
	Q_f[2, 2] = 50
	Q_f[3, 3] = 10
	system.set_final_cost(Q_f)
	system.set_goal(np.array([2, 4, np.pi/2, 0]))

	solver = CDDP(system, np.zeros(4), horizon=100)
	constraint = CircleConstraintForCar(np.ones(2), 0.5, system)
	constraint2 = CircleConstraintForCar(np.array([2, 2]), 1.0, system)
	for i in range(10):
		solver.backward_pass()
		solver.forward_pass()
	solver.add_constraint(constraint)
	solver.add_constraint(constraint2)
	system.set_goal(np.array([3, 3, np.pi/2, 0]))
	for i in range(20):
		solver.backward_pass()
		solver.forward_pass()
	solver.system.draw_trajectories(solver.x_trajectories)