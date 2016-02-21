'''
This is part of supporting code for <A Mixed Finite Element Solver for 
Natural Convection in Porous Media Using Automated Solution Techniques>.
Author: Chao Zhang (chao.zhang@auckland.ac.nz)

In order to produce the data used in the paper, please run

'mms_convergence.py'    for    Figure 1, Figure 2 and Table 1
'hrl.py'                for    Figure 4, Figure 5 and Table 2
'hfs.py                 for    Figure 7, Figure 8 and Table 21
'elder.py               for    Figure 10
'''

from convection import *

nxs = [32, 48, 64]
step_length = 5
Ras = range(50, 150, step_length)
T = 1.0e8
cfl = 0.2
for nx in nxs:
    dt = 1.0e-3
    if nx == 64:
        dt = dt/2
    for Ra in Ras:
        myProblem = Convection('hfs', Ra, nx, nx, T, dt, cfl)
        myProblem.generate_mesh()
        myProblem.mark_boundaries()
        myProblem.generate_function_spaces()
        myProblem.generate_functions(Ras[0], step_length)
        myProblem.define_boundary_expressions()
        myProblem.define_weakform()
        myProblem.define_boundary_condition()
        myProblem.create_variational_problem_and_solver()
        dt = myProblem.simulate()