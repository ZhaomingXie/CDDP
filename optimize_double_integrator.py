from systems import DoubleIntegrator, Car
from constraints import CircleConstraintForDoubleIntegrator, CircleConstraintForCar
from cddp import CDDP
import numpy as np


if __name__ == '__main__':
	system = DoubleIntegrator()
	system.set_cost(np.zeros((4, 4)), 0.05*np.identity(2))
	Q_f = np.identity(4)
	Q_f[0, 0] = 50
	Q_f[1, 1] = 50
	Q_f[2, 2] = 10
	Q_f[3, 3] = 10
	system.set_final_cost(Q_f)

	solver = CDDP(system, np.zeros(4), horizon=300)

	#solve for initial trajectories
	system.set_goal(np.array([0, 3, 0, 0]))
	solver.backward_pass()
	solver.forward_pass()
	
	constraint = CircleConstraintForDoubleIntegrator(np.ones(2), 0.5, system)
	constraint2 = CircleConstraintForDoubleIntegrator(np.array([1.5, 2.2]), 0.5, system)
	solver.add_constraint(constraint)
	solver.add_constraint(constraint2)
	system.set_goal(np.array([3, 3, 0, 0]))
	for i in range(30):
		solver.backward_pass()
		solver.forward_pass()
	solver.system.draw_trajectories(solver.x_trajectories)