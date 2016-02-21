from dolfin import *
from math import log as ln
set_log_level(ERROR)

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

def calerrornorm(u_e, u, Ve):
    u_Ve = interpolate(u, Ve)
    u_e_Ve = interpolate(u_e, Ve)
    e_Ve = Function(Ve)
    # Subtract degrees of freedom for the error field
    e_Ve.vector()[:] = u_e_Ve.vector().array() - \
                       u_Ve.vector().array()
    error = e_Ve**2*dx
    return sqrt(assemble(error))

if __name__ == '__main__':
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
            problem = MMS_convection(nx, flow_degree[case_i], heat_degree[case_i])
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