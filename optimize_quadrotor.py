from systems import Quadrotor
from constraints import SphereConstraintForQuadrotor
from cddp_quadrotor import CDDP
import numpy as np


if __name__ == '__main__':
	system = Quadrotor()
	system.set_cost(np.zeros((12, 12)), 0.02*np.identity(4))
	Q_f = np.identity(12)
	Q_f[0, 0] = 50
	Q_f[1, 1] = 50
	Q_f[2, 2] = 50
	# Q_f[6, 6] = 2
	# Q_f[7, 7] = 2
	# Q_f[8, 8] = 2
	Q_f[3, 3] = 10
	Q_f[4, 4] = 10
	Q_f[5, 5] = 10
	system.set_final_cost(Q_f)

	initial_state = np.array([-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	system.set_goal(np.array([-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
	solver = CDDP(system, initial_state, horizon=200)
	solver.forward_pass()

	constraint = SphereConstraintForQuadrotor(np.zeros(3), 2, system)
	solver.add_constraint(constraint)
	

	#solve for initial trajectories
	system.set_goal(np.array([-0.5, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
	solver.best_J = 100000000
	for i in range(20):
		solver.backward_pass()
		solver.forward_pass()

	print("initial trajectory generated")
	print(solver.x_trajectories[:, 199])
	solver.best_J = 100000000
	system.set_goal(np.array([2.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
	solver.reg_factor = 0.1
	solver.reg_factor_u = 0.1
	for i in range(200):
		solver.backward_pass()
		solver.forward_pass()
		print(solver.x_trajectories[:, 199])
	# solver.system.draw_trajectories(solver.x_trajectories)