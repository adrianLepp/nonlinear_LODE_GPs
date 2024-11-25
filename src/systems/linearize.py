
from sage.all import *


class Parameter():
            u = 1.2371e-4 * 0.1
            c13= 2.5046e-5
            c32= 2.5046e-5
            c2R= 1.9988e-5
            g= 9.81 
            A= 0.0154

param = Parameter()

#u_e = param.u * 0.1

# dx0 = 1/param.A*(u-param.c13*sign(x-z)*sqrt(2*param.g*abs(x-z))) #param.u is u
# dx1 = 1/param.A*(param.c32*sign(z-y)*sqrt(2*param.g*abs(z-y))-param.c2R*sqrt(2*param.g*abs(y)))
# dx2 = 1/param.A*(param.c13*sign(x-z)*sqrt(2*param.g*abs(x-z))-param.c32*sign(z-y)*sqrt(2*param.g*abs(z-y)))


# dx0 = 1/param.A*(param.u-param.c13*sign(x-z)*sqrt(2*param.g*abs(x-z))) #param.u is u
# dx1 = 1/param.A*(param.c32*sign(z-y)*sqrt(2*param.g*abs(z-y))-param.c2R*sqrt(2*param.g*abs(y)))
# dx2 = 1/param.A*(param.c13*sign(x-z)*sqrt(2*param.g*abs(x-z))-param.c32*sign(z-y)*sqrt(2*param.g*abs(z-y)))


# %% https://ask.sagemath.org/question/8557/solve-a-simple-system-of-non-linear-equations/


# R = QQ['x, y, z']
# (x, y, z,) = R._first_ngens(3)

#R = QQ['x, y, z']; (x,y,z) = R._first_ngens(3)
#GF(Integer(5))['x, y, z']


# F1 = symbolic_expression(1/param.A*(u_e-param.c13*sqrt(2*param.g*(x-z))))
# F2 = symbolic_expression(1/param.A*(param.c32*sqrt(2*param.g*(z-y))-param.c2R*sqrt(2*param.g*(y))))
# F3 = symbolic_expression(1/param.A*(param.c13*sqrt(2*param.g*(x-z))-param.c32*sqrt(2*param.g*(z-y))))


x1, x2, x3, u = var('x1, x2, x3, u', domain='positive')#domain='real'

f1 = symbolic_expression(1/param.A*(u-param.c13*sqrt(2*param.g*(x1-x3)))).function(x1,x3,u)
f2 = symbolic_expression(1/param.A*(param.c32*sqrt(2*param.g*(x3-x2))-param.c2R*sqrt(2*param.g*(x2)))).function(x2,x3)
f3 = symbolic_expression(1/param.A*(param.c13*sqrt(2*param.g*(x1-x3))-param.c32*sqrt(2*param.g*(x3-x2)))).function(x1,x2,x3)
states = [x1, x2, x3]
control = [u]
system_eqations = [f1, f2, f3] 



constraints = [
    #   x > z, 
    #   z > y, 
    #   x > 0, 
    #   y > 0, 
    #   z > 0
    ] 

#constraint = x1 >=x3

#eqn = [1/param.A*(u_e-param.c13*sqrt(2*param.g*abs(x-z))) ==0, 1/param.A*(param.c32*sqrt(2*param.g*abs(z-y))-param.c2R*sqrt(2*param.g*abs(y)))==0,  1/param.A*(param.c13*sqrt(2*param.g*abs(x-z))-param.c32*sqrt(2*param.g*abs(z-y))) ==0]

#constraints + 
#s = solve(eqn, [x, y, z]) 

#eqn.append(constraints)
#eqn2 = [F1==0, F2==0, F3==0]
# eqn2 = [f1==0, f2==0, f3==0]
# eqn3 = [f1, f2, f3]
#eqn3 = [F1, F2, F3]
#eqn2.append(constraints)

#s = solve(eqn2, [x, y, z])


def find_equilibrium(equations, variables, constraints=None):
    if constraints is not None:
        equations.append(constraints)
    return solve(equations, variables)

def filter_solutions(solutions):
    valid_solutions = []

    for sol in solutions:
        values = [float(sol[i].rhs()) for i in range(len(sol))]
        #if all(value > 0 for value in values):
        if values[0] > values[2] and values[2] > values[1] and values[1] > 0: #TODO: how can i make this constraint generic as variable
            valid_solutions.append(values)
    return valid_solutions

def get_system_matrices(equations, variables, states):
    A = jacobian(equations, variables)
    #print(J)
    return A
    #return J(x1=states[0], x2=states[1], x3=states[2])
     


def get_equilibrium_equations(system_equations,states, control):
    eqn = []
    for i in range(len(system_equations)):
        eqn.append(system_equations[i] == 0)

    solutions = find_equilibrium(eqn, states)
    solution = solutions[0]

    A = jacobian(system_eqations, states)
    b = jacobian(system_eqations, control)

    return solution, A, b

def solve_for_equilibrium(A, b, equilibrium, solution):
    for i in range(len(solution)):
        equilibrium[f'x{i+1}'] = float(solution[i](u=equilibrium['u']).rhs())
        
    #print(equilibrium)
    A_e = A(**equilibrium)
    b_e = b(**equilibrium)
    #A_e = A_e.apply_map(lambda x: x.simplify_rational())
    #b_e = b_e.apply_map(lambda x: x.simplify_rational())

    return A_e, b_e

    equilibrium ={
        'u': param.u * 0.3,
    }


#print(A(**equilibrium))#x1=x_e[0], x2=x_e[1], x3=x_e[2],

#valid_solutions = filter_solutions(solutions)
#print(valid_solutions)
#J = get_system_matrices(eqn3, variables, valid_solutions[0])
#J = get_system_matrices(eqn3, variables, solutions[0])

#equilibrium = u_e = param.u * 0.1
#print(J(u_e = param.u * 0.1))

# constraint 1: all values must be positive
# constraint 2: x > z, z > y, 


#print(valid_solutions)


#TODO: solve with constraints. only one solution should be valid

#print(latex(s))

#TODO: check out from scipy import optimize

'''
https://ask.sagemath.org/question/7883/solve-multidimensional-nonlinear-system-with-scipy/
https://ask.sagemath.org/question/8546/numerical-solution-of-a-system-of-non-linear-equations/
https://ask.sagemath.org/question/64965/algebraically-solving-system-of-nonlinear-equations-with-parameters/
https://ask.sagemath.org/question/10885/solving-non-linear-system-of-equations/
https://ask.sagemath.org/question/40063/how-to-solve-a-nonlinear-equation-system-numerically/

https://doc.sagemath.org/html/en/constructions/

'''

# %% calculate jacobian

# J = jacobian(eqn3, [x, y, z])
# for i in range(3):
#     for j in range(3):
#         print(J[i,j])

# # print(J)

# print(J(valid_solutions[0][0],valid_solutions[0][1],valid_solutions[0][2]))
# %%
