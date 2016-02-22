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

from dolfin import *
from Boundary_and_expressions import *
import numpy as np
set_log_level(ERROR)

class Convection(object):
    """Convection constructor"""
    def __init__(self, Ra, nx, ny, T, dt, cfl, floworder=0, heatorder=1):
        self.Ra = Ra
        self.nx = nx
        self.ny = ny
        self.T = T
        self.dt_init = dt
        self.t = dt
        self.dt = dt
        self.cfl = cfl
        self.floworder = floworder
        self.heatorder = heatorder
        self.steady_tol = 1.0e-5
        output_handle = file('nusselt_ra%d_nx%d.txt' % (self.Ra, self.nx), 'w')
        output_handle.close()
        output_handle = file('diff_ra%d_nx%d.txt' % (self.Ra, self.nx), 'w')
        output_handle.close()

    def generate_mesh(self):
        self.mesh = UnitSquareMesh(self.nx, self.nx)

    def mark_boundaries(self):
        self.boundaries = FacetFunction('size_t', self.mesh, 0)
        self.top = TopBoundary()
        self.top.mark(self.boundaries, 4)
        self.bottom = BottomBoundary()
        self.bottom.mark(self.boundaries, 2)
        self.right = RightBoundary()
        self.right.mark(self.boundaries, 3)
        self.left = LeftBoundary()
        self.left.mark(self.boundaries, 1)

    def generate_function_spaces(self):
        self.VEL = VectorFunctionSpace(self.mesh, "CG", 1)
        BDM = FunctionSpace(self.mesh, "BDM", self.floworder+1)
        DG0 = FunctionSpace(self.mesh, "DG", self.floworder)
        DG1 = FunctionSpace(self.mesh, "DG", self.heatorder)
        self.W = MixedFunctionSpace([BDM, DG0, DG1])

    def generate_functions(self, start_Ra=50, step_length=5):
        if self.Ra == start_Ra:
            self.w = Function(self.W)
            self.w0= Function(self.W)
        else:
            self.w = Function(self.W, 'steady_state_ra%d_nx%d.xml' % (Ra-step_length, nx))
            self.w0= Function(self.W, 'steady_state_ra%d_nx%d.xml' % (Ra-step_length, nx))
        self.u, self.p, self.ut = split(self.w)

    def define_boundary_expressions(self):
        pass

    def define_weakform(self):
        u, p, ut = split(self.w)
        u0, p0, ut0 = split(self.w0)
        v, q, vt = TestFunctions(self.W)
        ds = Measure("ds")[self.boundaries]
        # normal direction
        n = FacetNormal(self.mesh)
        F  = (dot(u, v) - div(v)*p - inner(self.gravity(ut), v) )*dx
        # Fluid mass conservation
        F += nabla_div(u)*q*dx
        # Heat transfer
        h = CellSize(self.mesh)
        alpha = Constant(5.0)
        if self.heatorder == 2:
            alpha = Constant(100.0)
        kappa = Constant(1.0)
        un = (dot(u, n) + abs(dot(u, n)))/2.0
        F_timegradient = (ut - ut0)*vt*dx
        # internal
        a_int = dot(grad(vt), kappa*grad(ut) - u*ut)*dx 
        #facet
        a_fac = kappa('+')*(alpha('+')/h('+'))*dot(jump(vt, n), jump(ut, n))*dS \
              - kappa('+')*dot(avg(grad(vt)), jump(ut, n))*dS \
              - kappa('+')*dot(jump(vt, n), avg(grad(ut)))*dS
        #velocity
        a_vel = dot(jump(vt), un('+')*ut('+') - un('-')*ut('-') )*dS + dot(vt, un*ut)*ds

        a = a_int + a_fac + a_vel
        F_heat = F_timegradient + self.dt*a
        F += F_heat
        self.F = F

    def create_variational_problem_and_solver(self):
        dw = TrialFunction(self.W)
        J = derivative(self.F, self.w, dw)
        self.problem = NonlinearVariationalProblem(self.F, self.w, self.bc, J)
        self.solver = NonlinearVariationalSolver(self.problem)
        self.solver.parameters["newton_solver"]["linear_solver"] = "gmres"

    def gravity(self, ut):
        val = as_vector([0.0, self.Ra*ut])
        return val

    def define_boundary_condition(self):
        pass

    def simulate(self, plot_slu=False):
        if plot_slu:
            fig_u = plot(self.w.sub(0), axes=True, title = 'Velocity')
            fig_p = plot(self.w.sub(1), axes=True, title = 'Pressure')
            fig_ut = plot(self.w.sub(2), axes=True, title = 'Temperature')
        self.dt_cfl = self.dt
        count = 1
        while self.t < self.T:
            print '----------------------'
            print '     Ra = %d' % self.Ra
            print '     nx = %d' % self.nx
            print '  count = %d' % count
            print '      t = %.6E' % self.t
            print '     dt = %.6E' % self.dt
            print ' dt_cfl = %.6E' % self.dt_cfl
            self.ut_bc.t = self.t
            self.solver.solve()
            if self.check_steady_state():
                print 'Steady state reached'
                break
            self.calculate_nusselt()
            self.w0.vector()[:] = self.w.vector()
            if plot_slu:
                fig_u.plot()
                fig_p.plot()
                fig_ut.plot()
            self.determine_dt()
            self.t += self.dt
            count += 1
        File('steady_state_ra%d_nx%d.xml' % (self.Ra, self.nx)) << self.w 

    def check_steady_state(self):
        is_steady_state = False
        warray = self.w.vector().array()
        w0array= self.w0.vector().array()
        l2norm = np.linalg.norm((warray - w0array))
        output = 'L2 Norm = %.6E' % l2norm
        print output
        output_handle = file('diff_ra%d_nx%d.txt' % (self.Ra, self.nx), 'a')
        output_handle.write('%.6E,%.6E\n' % (self.t, l2norm))
        output_handle.close()
        if l2norm < self.steady_tol:
            is_steady_state = True
        return is_steady_state

    def determine_dt(self):
        velo = interpolate(self.w.sub(0), self.VEL)
        max_velocity = np.max(np.abs(velo.vector().array()))
        hmin = self.mesh.hmin()
        self.dt_cfl = self.cfl*hmin/max_velocity
        if self.dt_cfl < self.dt_init:
            self.dt = self.dt_cfl
        else:
            self.dt = self.dt_init

    def calculate_nusselt(self):

        grad_norm = inner(grad(self.ut),grad(self.ut))*dx
        nusselt = assemble(grad_norm)
        output = "Nusselt = %.6E\n" % nusselt
        print output
        output_handle = file('nusselt_ra%d_nx%d.txt' % (self.Ra, self.nx), 'a')
        output_handle.write('%.6E,%.6E\n' % (self.t, nusselt))
        output_handle.close()
        return nusselt

    def solve(self):
        self.generate_mesh()
        self.mark_boundaries()
        self.generate_function_spaces()
        self.generate_functions()
        self.define_boundary_expressions()
        self.define_weakform()
        self.define_boundary_condition()
        self.create_variational_problem_and_solver()
        self.simulate()
        return self.dt

class HRL(Convection):
    """docstring for HRL"""
    def define_boundary_expressions(self):
        self.ut_bc = hrl_ut_bc_expression(self.t, self.Ra)
        self.zero_normal_flux_bc = zero_normal_flux_bc_expression(self.mesh)

    def define_boundary_condition(self):
        bc1 = [DirichletBC(self.W.sub(0), self.zero_normal_flux_bc, AllBoundary())]

        bc2 = [DirichletBC(self.W.sub(2), self.ut_bc, self.boundaries, 2, "geometric"), \
               DirichletBC(self.W.sub(2), self.ut_bc, self.boundaries, 4, "geometric")]
        self.bc = bc1 + bc2

class HFS(Convection):
    """docstring for HFS"""
    def define_boundary_expressions(self):
        self.ut_bc = hfs_ut_bc_expression(self.t)
        self.zero_normal_flux_bc = zero_normal_flux_bc_expression(self.mesh)

    def define_boundary_condition(self):
        bc1 = [DirichletBC(self.W.sub(0), self.zero_normal_flux_bc, AllBoundary())]

        bc2 = [DirichletBC(self.W.sub(2), self.ut_bc, self.boundaries, 1, "geometric"), \
               DirichletBC(self.W.sub(2), self.ut_bc, self.boundaries, 3, "geometric")]
        self.bc = bc1 + bc2

class Elder(Convection):
    """docstring for Elder"""
    def __init__(self, Ra, nx, ny, T, dt, cfl, floworder=0, heatorder=1):
        self.Ra = Ra
        self.nx = nx
        self.ny = ny
        self.T = T
        self.dt_init = dt
        self.t = dt
        self.dt = dt
        self.cfl = cfl
        self.floworder = floworder
        self.heatorder = heatorder
        self.steady_tol = 1.0e-5

    def generate_mesh(self):
        self.mesh = UnitSquareMesh(self.nx, self.ny)
        self.mesh.coordinates()[:,0] = self.mesh.coordinates()[:,0]*2

    def mark_boundaries(self):
        self.boundaries = FacetFunction('size_t', self.mesh, 0)
        self.top = TopBoundary()
        self.top.mark(self.boundaries, 2)
        self.centralbottom = CentralBottom()
        self.centralbottom.mark(self.boundaries, 1)
        self.edgebottom = EdgeBottom()
        self.edgebottom.mark(self.boundaries, 3)

    def generate_functions(self):
        self.w = Function(self.W)
        self.w0= Function(self.W)

    def define_boundary_expressions(self):
        self.ut_bc = ut_bc_expression(self.t)
        self.zero_normal_flux_bc = zero_normal_flux_bc_expression(self.mesh)

    def define_boundary_condition(self):
        bc1 = [DirichletBC(self.W.sub(0), self.zero_normal_flux_bc, AllBoundary())]
        bc2 = [DirichletBC(self.W.sub(2), self.ut_bc, self.boundaries, 1, "geometric"), \
               DirichletBC(self.W.sub(2), self.ut_bc, self.boundaries, 2, "geometric"), \
               DirichletBC(self.W.sub(2), Constant(0.0), self.boundaries, 3, "geometric")]
        self.bc = bc1 + bc2

    def simulate(self, plot_slu=False):
            if plot_slu:
                fig_u = plot(self.w.sub(0), axes=True, title = 'Velocity')
                fig_p = plot(self.w.sub(1), axes=True, title = 'Pressure')
                fig_ut = plot(self.w.sub(2), axes=True, title = 'Temperature')
            self.dt_cfl = self.dt
            count = 1
            while self.t < self.T:
                print '----------------------'
                print '     Ra = %d' % self.Ra
                print '     nx = %d' % self.nx
                print '  count = %d' % count
                print '      t = %.6E' % self.t
                print '     dt = %.6E' % self.dt
                print ' dt_cfl = %.6E' % self.dt_cfl
                self.ut_bc.t = self.t
                self.solver.solve()
                self.w0.vector()[:] = self.w.vector()
                if plot_slu:
                    fig_u.plot()
                    fig_p.plot()
                    fig_ut.plot()
                self.determine_dt()
                if (count%(10) < 0.01):
                    File('solutions/transient_solution_at'+ '%.6f' %t + '.xml') << w
                self.t += self.dt
                count += 1

class MMS_convection(object):
    """MMS_convection constructor"""
    def __init__(self, nx, floworder=0, heatorder=1):
        self.nx = nx
        self.floworder = floworder
        self.heatorder = heatorder

    def generate_mesh(self):
        self.mesh = UnitSquareMesh(self.nx, self.nx)

    def mark_boundaries(self):
        self.boundaries = FacetFunction('size_t', self.mesh, 0)
        self.top = TopBoundary()
        self.top.mark(self.boundaries, 4)
        self.bottom = BottomBoundary()
        self.bottom.mark(self.boundaries, 2)
        self.right = RightBoundary()
        self.right.mark(self.boundaries, 3)
        self.left = LeftBoundary()
        self.left.mark(self.boundaries, 1)

    def generate_function_spaces(self):
        self.CG  = FunctionSpace(self.mesh, "CG", self.heatorder+1)
        self.V_CG = VectorFunctionSpace(self.mesh, "CG", self.heatorder+1)
        BDM = FunctionSpace(self.mesh, "BDM", self.floworder+1)
        DG0 = FunctionSpace(self.mesh, "DG", self.floworder)
        DG1 = FunctionSpace(self.mesh, "DG", self.heatorder)
        self.W = MixedFunctionSpace([BDM, DG0, DG1])

    def generate_functions(self):
        self.w = Function(self.W)
        self.w0= Function(self.W)
        (self.u, self.p, self.ut) = split(self.w)
        (self.u0, self.p0, self.ut0) = split(self.w0)
        (self.v, self.q, self.vt) = TestFunctions(self.W)

    def define_boundary_expressions(self):
        self.p_bc = mms_p_bc_expression()
        self.ut_bc = mms_ut_bc_expression()
        self.u_bc = mms_u_bc_expression()

    def define_weakform(self):
        self.Ra = 1
        f_1 = Expression('sin(x[0]) + cos(x[1])')
        f_2 = Expression(('0.0', '-sin(x[0]) - sin(x[1])'))
        f_3 = Expression('-cos(x[0])*cos(x[0]) + sin(x[1])*cos(x[1]) + sin(x[0]) + sin(x[1])')
        ds = Measure("ds")[self.boundaries]
        n = FacetNormal(self.mesh)
        self.F = (dot(self.u, self.v) - div(self.v)*self.p - inner(self.gravity(self.ut), self.v) - inner(f_2, self.v) )*dx + dot(n, self.v)*self.p_bc*ds
        self.F += nabla_div(self.u)*self.q*dx - f_1*self.q*dx
        h = CellSize(self.mesh)
        if self.heatorder == 1:
            alpha = Constant(5.0)
        else:
            alpha = Constant(100.0)
        kappa = Constant(1.0)
        un = (dot(self.u, n) + abs(dot(self.u, n)))/2.0
        a_int = dot(grad(self.vt), kappa*grad(self.ut) - self.u*self.ut)*dx - self.ut*self.vt*nabla_div(self.u)*dx
        a_fac = kappa('+')*(alpha('+')/h('+'))*dot(jump(self.vt, n), jump(self.ut, n))*dS \
                - kappa('+')*dot(avg(grad(self.vt)), jump(self.ut, n))*dS \
                - kappa('+')*dot(jump(self.vt, n), avg(grad(self.ut)))*dS
        a_vel = dot(jump(self.vt), un('+')*self.ut('+') - un('-')*self.ut('-') )*dS + dot(self.vt, un*self.ut)*ds
        self.F += a_int + a_fac + a_vel - f_3*self.vt*dx

    def define_boundary_condition(self):
        self.bc = [DirichletBC(self.W.sub(2), self.ut_bc, self.top, "geometric"),\
                   DirichletBC(self.W.sub(2), self.ut_bc, self.bottom, "geometric"),\
                   DirichletBC(self.W.sub(2), self.ut_bc, self.right, "geometric"),\
                   DirichletBC(self.W.sub(2), self.ut_bc, self.left, "geometric")]

    def create_variational_problem_and_solver(self):
        dw = TrialFunction(self.W)
        J = derivative(self.F, self.w, dw)
        self.problem = NonlinearVariationalProblem(self.F, self.w, self.bc, J)
        self.solver = NonlinearVariationalSolver(self.problem)
        self.solver.parameters["newton_solver"]["linear_solver"] = "gmres"

    def gravity(self, ut):
        val = as_vector([0.0, self.Ra*ut])
        return val

    def calculate_error(self):
        File(str(self.nx)+'solution.xml') << self.w 
        (u, p, ut) = self.w.split()
        p_error_norm = calerrornorm(self.p_bc, p, self.CG)
        ut_error_norm = calerrornorm(self.ut_bc, ut, self.CG)
        u_error_norm = calerrornorm(self.u_bc, u, self.V_CG)
        return [p_error_norm, u_error_norm, ut_error_norm]

    def solve(self):
        self.generate_mesh()
        self.mark_boundaries()
        self.generate_function_spaces()
        self.generate_functions()
        self.define_boundary_expressions()
        self.define_weakform()
        self.define_boundary_condition()
        self.create_variational_problem_and_solver()
        self.solver.solve()
