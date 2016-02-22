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

class AllBoundary(SubDomain):
     def inside(self, x, on_boundary):
        return on_boundary

class TopBoundary(SubDomain):
     def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)

class BottomBoundary(SubDomain):
     def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
       return on_boundary and near(x[0], 1.0)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
       return on_boundary and near(x[0], 0.0)

class CornerBottomBoundary(SubDomain):
     def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0) and x[0]>1.0

class CentralBottomBoundary(SubDomain):
     def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0) and (x[0]<=1.0)

class mms_p_bc_expression(Expression):
    def eval(self, values, x):
        values[0] = sin(x[0]) + cos(x[1])

class mms_ut_bc_expression(Expression):
    def eval(self, values, x):
        values[0] = sin(x[0]) + sin(x[1])

class mms_u_bc_expression(Expression):
    def eval(self, values, x):
        values[0] = -cos(x[0])
        values[1] = sin(x[1])
    def value_shape(self):
        return (2,)

class hfs_ut_bc_expression(Expression):
        def __init__(self, t):
            self.t = t
        def eval(self, values, x):
            values[0] = (1.0 - x[0])*1.0

class hrl_ut_bc_expression(Expression):
        def __init__(self, t, Ra):
            self.t = t
            self.Ra = Ra
        def eval(self, values, x):
            if x[1] == 1:
                values[0] = 0.0
            else:
                if self.Ra == 50:
                    if self.t < 1.0e-2:
                        values[0] = 1.0 + 1.0e-6*sin(2*pi*x[0])
                    else:
                        values[0] = 1.0
                else:
                    values[0] = (1.0 - x[1])*1.0

class zero_normal_flux_bc_expression(Expression):
        def __init__(self, mesh):
            self.mesh = mesh
        def eval_cell(self, values, x, ufc_cell):
            cell = Cell(self.mesh, ufc_cell.index)
            n = cell.normal(ufc_cell.local_facet)
            g = 0.0
            values[0] = g*n[0]
            values[1] = g*n[1]
        def value_shape(self):
            return (2,)

class TopBoundary(SubDomain):
     def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)

class EdgeBottom(SubDomain):
     def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0) and x[0]>1.0

class CentralBottom(SubDomain):
     def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0) and (x[0]<=1.0)

class ut_bc_expression(Expression):
        def __init__(self, t):
            self.t = t
        def eval(self, values, x):
            values[0] = (1.0 - x[1])*1.0

def calerrornorm(u_e, u, Ve):
    u_Ve = interpolate(u, Ve)
    u_e_Ve = interpolate(u_e, Ve)
    e_Ve = Function(Ve)
    # Subtract degrees of freedom for the error field
    e_Ve.vector()[:] = u_e_Ve.vector().array() - \
                       u_Ve.vector().array()
    error = e_Ve**2*dx
    return sqrt(assemble(error))
