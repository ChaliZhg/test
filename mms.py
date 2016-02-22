'''
This is part of supporting code for 'A Mixed Finite Element Solver for 
Natural Convection in Porous Media Using Automated Solution Techniques'.
Author: Chao Zhang (chao.zhang@auckland.ac.nz)

In order to produce the data used in the paper, please run

'mms.py'     for    Figure 1, Figure 2 and Table 1
'hrl.py'     for    Figure 4, Figure 5 and Table 2
'hfs.py      for    Figure 7, Figure 8 and Table 21
'elder.py    for    Figure 10
'''

from dolfin import *
from convection import MMS
from boundaries_and_expressions import *
from math import log as ln
set_log_level(ERROR)

output_handle = file('convergence.txt', 'w')
flow_degree = [0,1]
heat_degree = [1,2]
# The following script is adjust from FEniCS fundamental demo at
# http://fenicsproject.org/documentation/tutorial/fundamentals.html#computing-functionals
for case_i in range(len(flow_degree)):
    output = '[[%d]]\nApproximation order: pressure=%d, velocity=%d, temperature=%d\n'\
        %(case_i, flow_degree[case_i], flow_degree[case_i]+1, heat_degree[case_i])
    output_handle.write(output)
    h = []  # to store element sizes
    E = []  # to store errors
    nxs = [4, 8, 16, 32, 64, 128]
    for nx in nxs:
        print "solving case nx = [%d]" % nx
        output_handle.write( "solving case nx = [%d]\n" % nx )
        h.append(1.0/nx)
        problem = MMS(nx, flow_degree[case_i], heat_degree[case_i])
        problem.solve()
        E.append(problem.calculate_error())

    print 'Calculate convergence rate'
    terms = ['Pressure', 'Velocity', 'Temperature']
    for i in range(3):
        print "[%d] Convergence in %s" % (i, terms[i])
        output_handle.write("[%d] Convergence in %s\n" % (i, terms[i]))
        for j in range(1, len(E)):
            r = ln(E[j][i]/E[j-1][i])/ln(h[j]/h[j-1])
            output = 'h=1/%3d error=%8.2E rate=%.2f\n' % (nxs[j], E[j][i], r)
            print output
            output_handle.write(output)
output_handle.close()
print "DONE"
