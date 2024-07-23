from time import time
import numpy as np

# from Models.diffdrive_2d import Model
# from Models.diffdrive_2d_plot import plot
from Models.rocket_landing_3d_experimental import Model
from Models.rocket_landing_3d_plot import plot

"""
Python implementation of the Successive Convexification algorithm.

Rocket trajectory model based on the
'Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time' paper
by Michael Szmuk and Behçet Açıkmeşe.

Implementation by Sven Niederberger (s-niederberger@outlook.com)
"""

def single_check(r_I_init, v_I_init, disp = False): 
    from FreeFinalTime.parameters import K, iterations, w_nu, w_sigma, tr_radius, verbose_solver, solver, rho_0, rho_1, rho_2, alpha, beta
    from FreeFinalTime.discretization import FirstOrderHold
    from FreeFinalTime.scproblem import SCProblem
    from utils import format_line, save_arrays

    m = Model(r_I_init = r_I_init, v_I_init = v_I_init)
    m.nondimensionalize()

    # state and input
    X = np.empty(shape=[m.n_x, K])
    U = np.empty(shape=[m.n_u, K])

    # INITIALIZATION--------------------------------------------------------------------------------------------------------
    sigma = m.t_f_guess
    X, U = m.initialize_trajectory(X, U)

    # START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
    all_X = [m.x_redim(X.copy())]
    all_U = [m.u_redim(U.copy())]
    all_sigma = [sigma]

    integrator = FirstOrderHold(m, K)
    problem = SCProblem(m, K)

    last_nonlinear_cost = None
    converged = False
    for it in range(iterations):
        t0_it = time()
        if disp:
            print('-' * 50)
        print('-' * 18 + f' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)
        if disp:
            print('-' * 50)

        t0_tm = time()
        A_bar, B_bar, C_bar, S_bar, z_bar = integrator.calculate_discretization(X, U, sigma)
        if disp:
            print(format_line('Time for transition matrices', time() - t0_tm, 's'))

        problem.set_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                               X_last=X, U_last=U, sigma_last=sigma,
                               weight_nu=w_nu, weight_sigma=w_sigma, tr_radius=tr_radius)
        
        while True:
            if solver == 'MOSEK':
                error = problem.solve(verbose=verbose_solver, solver=solver)
            else:
                error = problem.solve(verbose=verbose_solver, solver=solver, max_iters=200)
            if disp:
                print(format_line('Solver Error', error))
            if error and problem.get_variable('X') is None:
                print('Failed to solve problem. Exiting.')
                return None, None, None, False, None, None, None, None
            
            # get solution
            new_X = problem.get_variable('X')
            new_U = problem.get_variable('U')
            new_sigma = problem.get_variable('sigma')

            X_nl = integrator.integrate_nonlinear_piecewise(new_X, new_U, new_sigma)

            linear_cost_dynamics = np.linalg.norm(problem.get_variable('nu'), 1)
            nonlinear_cost_dynamics = np.linalg.norm(new_X - X_nl, 1)

            linear_cost_constraints = m.get_linear_cost()
            nonlinear_cost_constraints = m.get_nonlinear_cost(X=new_X, U=new_U)

            linear_cost = linear_cost_dynamics + linear_cost_constraints  # J
            nonlinear_cost = nonlinear_cost_dynamics + nonlinear_cost_constraints  # L

            if last_nonlinear_cost is None:
                last_nonlinear_cost = nonlinear_cost
                X = new_X
                U = new_U
                sigma = new_sigma
                break

            actual_change = last_nonlinear_cost - nonlinear_cost  # delta_J
            predicted_change = last_nonlinear_cost - linear_cost  # delta_L

            if disp:
                print('')
                print(format_line('Virtual Control Cost', linear_cost_dynamics))
                print(format_line('Constraint Cost', linear_cost_constraints))
                print('')
                print(format_line('Actual change', actual_change))
                print(format_line('Predicted change', predicted_change))
                print('')
                print(format_line('Final time', sigma))
                print('')

            if abs(predicted_change) < 1e-4:
                converged = True
                break
            else:
                rho = actual_change / predicted_change
                if rho < rho_0:
                    # reject solution
                    tr_radius /= alpha
                    if disp:
                        print(f'Trust region too large. Solving again with radius={tr_radius}')
                else:
                    # accept solution
                    X = new_X
                    U = new_U
                    sigma = new_sigma

                    if disp:
                        print('Solution accepted.')

                    if rho < rho_1:
                        if disp:
                            print('Decreasing radius.')
                        tr_radius /= alpha
                    elif rho >= rho_2:
                        if disp:
                            print('Increasing radius.')
                        tr_radius *= beta

                    last_nonlinear_cost = nonlinear_cost
                    break

            problem.set_parameters(tr_radius=tr_radius)

            if disp:
                print('-' * 50)
        if disp:
            print('')
            print(format_line('Time for iteration', time() - t0_it, 's'))
            print('')

        all_X.append(m.x_redim(X.copy()))
        all_U.append(m.u_redim(U.copy()))
        all_sigma.append(sigma)

        if converged and disp:
            print(f'Converged after {it + 1} iterations.')
            break

    all_X = np.stack(all_X)
    all_U = np.stack(all_U)
    all_sigma = np.array(all_sigma)
    if not converged:
        print('Maximum number of iterations reached without convergence.')

    # save trajectory to file for visualization
    # save_arrays('output/trajectory/', {'X': all_X, 'U': all_U, 'sigma': all_sigma})

    # plot trajectory
    plot(all_X, all_U, all_sigma)

    output = all_X, all_U, all_sigma, converged, linear_cost_dynamics, linear_cost_constraints, nonlinear_cost_dynamics, nonlinear_cost_constraints
    del m
    return output


if __name__ == '__main__':
    # _, _, _, converged, linear_cost_dynamics, linear_cost_constraints, nonlinear_cost_dynamics, nonlinear_cost_constraints = single_check(r_I_init = [100, 0,   100], v_I_init = [  0, 0, -90 ])
    # exit()
    r_v_combos = [{"r_I_init": [  0, 0, 100],
                   "v_I_init": [  0, 0, -50]}]
    for r_v_combo in r_v_combos:
        # try:
        output = single_check(r_I_init = r_v_combo["r_I_init"], v_I_init = r_v_combo["v_I_init"])
        all_X, all_U, all_sigma, converged, linear_cost_dynamics, linear_cost_constraints, nonlinear_cost_dynamics, nonlinear_cost_constraints = output
        print(f"Initial Position: {r_v_combo['r_I_init']}, Initial Velocity: {r_v_combo['v_I_init']}")
        print(f" Final Time: {all_sigma[-1]}")
        print(f" Converged: {converged}")
        print(f" Linear Cost Dynamics: {linear_cost_dynamics}")
        print(f" Linear Cost Constraints: {linear_cost_constraints}")
        print(f" Nonlinear Cost Dynamics: {nonlinear_cost_dynamics}")
        print(f" Nonlinear Cost Constraints: {nonlinear_cost_constraints}")
        # except:
        #     print(f"Failed for {r_v_combo}")
        #     continue