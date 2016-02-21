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
import numpy as np
set_log_level(ERROR)
# constant

def convection_solver(Ra, dt, nx, porder, plot_slu=False):
    # heat
    # Ra = 400

    height = 1.0

    # times
    # dt = 1.0e-5
    T  = 1.1e-1
    t  = dt
    count = 1

    # parameters["form_compiler"]["quadrature_degree"] = 2
    # nx = 40
    mesh = UnitSquareMesh(nx, nx)
    mesh.coordinates()[:,0] = mesh.coordinates()[:,0]*2
    boundaries = FacetFunction('size_t', mesh, 0)

    class TopBoundary(SubDomain):
         def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1.0)

    class CornerBoundary(SubDomain):
         def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0) and x[0]>1.0

    class BottomBoundary(SubDomain):
         def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0) and (x[0]<=1.0)

    TopBoundary().mark(boundaries, 2)
    CornerBoundary().mark(boundaries, 3)
    BottomBoundary().mark(boundaries, 1)

    # plot(mesh, interactive = True, title = 'mesh')
    # plot(boundaries, interactive = True, title = 'boundaries')

    # function space
    BDM = FunctionSpace(mesh, "BDM", porder+1)
    VEL = VectorFunctionSpace(mesh, 'CG', 1)
    DG0 = FunctionSpace(mesh, "DG", porder)
    DG1 = FunctionSpace(mesh, "DG", porder+1)
    W = MixedFunctionSpace([BDM, DG0, DG1])

    w = Function(W)
    w0= Function(W)

    # test functions
    (u, p, ut) = split(w)
    (u0, p0, ut0) = split(w0)
    (v, q, vt) = TestFunctions(W)

    ## display initial values
    # plot(u, title = 'Velocity', interactive = True)
    # plot(p, title = 'Pressure', interactive = True)
    # plot(ut, title = 'Temperature', interactive = True)

    # boundary condition classes # this is not supposed to be used here
    class p_bc_expression(Expression):
        def __init__(self, t):
            self.t = t
        def eval(self, values, x):
            values[0] = 0.0
    p_bc = p_bc_expression(t)

    class ut_bc_expression(Expression):
        def __init__(self, t):
            self.t = t
        def eval(self, values, x):
            values[0] = (1.0 - x[1])*1.0
    ut_bc = ut_bc_expression(t)

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
    zero_normal_flux_bc = zero_normal_flux_bc_expression(mesh)

    # define measure ds
    ds = Measure("ds")[boundaries]

    # normal direction
    n = FacetNormal(mesh)

    def gravity(ut):
        val = as_vector([0.0, Ra*ut])
        return val

    # Define variational form
    # Darcy
    F  = (dot(u, v) - div(v)*p - inner(gravity(ut), v) )*dx + dot(n, v)*p_bc*ds(3)

    # Fluid mass conservation
    F += nabla_div(u)*q*dx

    h = CellSize(mesh)
    h_avg = (h('+') + h('-'))/2
    alpha = Constant(5.0)
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
    F_heat = F_timegradient + dt*a
    F += F_heat

    # boundary condition
    def all_domain(x, on_boundary):
        return on_boundary

    # no flow boundary condition
    # bc1 = [DirichletBC(W.sub(0), zero_normal_flux_bc, all_domain)]
    bc1 = [DirichletBC(W.sub(0), zero_normal_flux_bc, all_domain)]
    bc2 = [DirichletBC(W.sub(2), ut_bc, boundaries, 1, "geometric"), \
           DirichletBC(W.sub(2), ut_bc, boundaries, 2, "geometric"), \
           DirichletBC(W.sub(2), Constant(0.0), boundaries, 3, "geometric")]


    bc = bc1 + bc2

    # problem and solver
    dw   = TrialFunction(W)
    J = derivative(F, w, dw)
    problem = NonlinearVariationalProblem(F, w, bc, J)
    solver  = NonlinearVariationalSolver(problem)

    solver.parameters["newton_solver"]["linear_solver"] = "gmres"
    # solver.parameters["newton_solver"]["preconditioner"] = "ilu"

    # plot during calculation
    if plot_slu:
        fig_u = plot(w.sub(0), axes=True, title = 'Velocity')
        fig_p = plot(w.sub(1), axes=True, title = 'Pressure')
        fig_ut = plot(w.sub(2), axes=True, title = 'Temperature')

    dt_cfl = dt
    while t < T:
        print '----------------------'
        print '     Ra = %d' % Ra
        print '     nx = %d' % nx
        print '  count = %d' % count
        print '      T = %.6E' % t
        print '     dt = %.6E' % dt
        print ' dt_cfl = %.6E' % dt_cfl
        p_bc.t = t
        ut_bc.t = t
        solver.solve()
        w0.vector()[:] = w.vector()
        if plot_slu:
            fig_u.plot()
            fig_p.plot()
            fig_ut.plot()

        # update dt
        velo = interpolate(w.sub(0), VEL)
        max_velocity = np.max(np.abs(velo.vector().array()))
        hmin = mesh.hmin()
        cfl = 0.5
        dt_cfl = cfl*hmin/max_velocity
        # print 'dt_cfl=',dt_cfl
        if dt_cfl < dt:
            dt = dt_cfl

        # save solution   
        if (count%(5) < 0.01):
            File('solutions/transient_solution_at'+ '%.6f' %t + '.xml') << w
        # list_timings()

        t += dt
        count += 1

    File('transient_solution.xml') << w 
    if plot_slu:
        interactive()

Ra = 400.0
dt = 1.0e-5
nx = 40
porder = 0
convection_solver(Ra, dt, nx, porder, True)
print "DONE"