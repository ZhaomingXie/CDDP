from systems import Quadrotor
from constraints import CircleConstraintForDoubleIntegrator, CircleConstraintForCar
from cddp_quadrotor import CDDP
import numpy as np


if __name__ == '__main__':
	system = Quadrotor()
	system.set_cost(np.zeros((12, 12)), 0.02*np.identity(4))
	Q_f = np.identity(12)
	Q_f[0, 0] = 50
	Q_f[1, 1] = 50
	Q_f[2, 2] = 50
	Q_f[3, 3] = 2
	Q_f[4, 4] = 2
	Q_f[5, 5] = 2
	system.set_final_cost(Q_f)

	initial_state = np.array([-3.5, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0])
	system.set_goal(np.array([-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
	solver = CDDP(system, initial_state, horizon=200)
	solver.forward_pass()
	

	#solve for initial trajectories
	system.set_goal(np.array([-2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
	solver.best_J = 100000000
	for i in range(10):
		solver.backward_pass()
		solver.forward_pass()

	# constraint = CircleConstraintForCar(np.ones(2), 0.5, system)
	# constraint2 = CircleConstraintForCar(np.array([2, 2]), 1.0, system)
	# solver.add_constraint(constraint2)
	# system.set_goal(np.array([3, 3, np.pi/2, 0]))
	for i in range(20):
		solver.backward_pass()
		solver.forward_pass()
	# solver.system.draw_trajectories(solver.x_trajectories)