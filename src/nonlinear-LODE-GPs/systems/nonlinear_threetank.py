from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
import torch
import numpy as np
from systems import ODE_System
from systems.linearize import solve_for_equilibrium, get_equilibrium_equations


# R = QQ['x, x1, x2, x3, u']
# (x, x1, x2, x3, u) = R._first_ngens(5)

R = QQ['x']; (x,) = R._first_ngens(1)

class Parameter():
            u = 1.2371e-4
            _u = 1
            c13= 2.5046e-5
            c32= 2.5046e-5
            c2R= 1.9988e-5
            g= 9.81 
            A= 0.0154

class Nonlinear_ThreeTank(ODE_System):
    def __init__(self):
        super().__init__(3,1)

        self.param = Parameter()

        x1, x2, x3, u = var('x1, x2, x3, u', domain='positive')#domain='real'

        f1 = symbolic_expression(1/self.param.A*(self.param._u * u-self.param.c13*sqrt(2*self.param.g*(x1-x3)))).function(x1,x3,u)
        f2 = symbolic_expression(1/self.param.A*(self.param.c32*sqrt(2*self.param.g*(x3-x2))-self.param.c2R*sqrt(2*self.param.g*(x2)))).function(x2,x3)
        f3 = symbolic_expression(1/self.param.A*(self.param.c13*sqrt(2*self.param.g*(x1-x3))-self.param.c32*sqrt(2*self.param.g*(x3-x2)))).function(x1,x2,x3)

        self.state_var = [x1, x2, x3]
        self.control_var = [u]
        self.system_eqations = [f1, f2, f3] 

        solution, A, b = get_equilibrium_equations(self.system_eqations, self.state_var, self.control_var)

        self.equilibrium_solution = solution
        self.A = A
        self.b = b

        # equilibrium ={
        # 'u': self.param.u * u_r_rel,
        # }
        # A_r, b_r = solve_for_equilibrium(A, b, equilibrium, solution)

        # self.A_r = A_r #.n() #matrix(R,A_r)
        # self.b_r =  b_r #.n() #matrix(R,b_r)

        # self.equilibrium = [ equilibrium['x1'],equilibrium['x2'],equilibrium['x3'],equilibrium['u']] 

        # u_r  = self.param.u*u_r_rel  
        # x_r1, x_r2, x_r3 = self.get_equilibrium(u_r)
        # print('equilibrium for nonlinear Threetank: ', x_r1, x_r2, x_r3)
        # self.equilibrium = [x_r1, x_r2, x_r3, u_r]
        # self.A_r, self.b_r = self.get_linearized_state_space(u_r, x_r1, x_r2, x_r3)

    def get_equilibrium(self,u_r):
        x_r1=(1+2*(self.param.c2R/self.param.c32)**2)*(1)/(2*self.param.g*self.param.c2R**2)*u_r**2
        x_r2=u_r**2/(2*self.param.g*self.param.c2R**2)
        x_r3=(1+(self.param.c2R/self.param.c32)**2)*(1)/(2*self.param.g*self.param.c2R**2)*u_r**2

        return x_r1, x_r2, x_r3  
    
    def get_linearized_state_space(self, u_r, x_r1, x_r2, x_r3):
        # first line
        a11=-self.param.c13*self.param.g/self.param.A*1/(sqrt(2*self.param.g*(x_r1-x_r3)))
        a12=0
        a13=self.param.c13*self.param.g/self.param.A*1/(sqrt(2*self.param.g*(x_r1-x_r3)))
        #second line
        a21=0
        a22=-self.param.c32*self.param.g/self.param.A*1/(sqrt(2*self.param.g*(x_r3-x_r2)))-(self.param.c2R*self.param.g)/self.param.A*1/(sqrt(2*self.param.g*x_r2))
        a23=self.param.c32*self.param.g/self.param.A*1/(sqrt(2*self.param.g*(x_r3-x_r2)))
        #third line
        a31=self.param.c13*self.param.g/self.param.A*1/(sqrt(2*self.param.g*(x_r1-x_r3)))
        a32=self.param.c32*self.param.g/self.param.A*1/(sqrt(2*self.param.g*(x_r3-x_r2)))
        a33=-self.param.c13*self.param.g/self.param.A*1/(sqrt(2*self.param.g*(x_r1-x_r3)))-self.param.c32*self.param.g/self.param.A*1/(sqrt(2*self.param.g*(x_r3-x_r2))) 

        A_r = [[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]]

        b_r=[1/self.param.A, 0, 0]
        return A_r, b_r
    
    def get_ODEmatrix(self, u_r_rel:float):
        #TODO: the matrices A b and Ix could surely be merged in a more elegant way
        #  a.exact_rational()
        #  a.nearby_rational(0.00001)

        equilibrium ={
        'u': self.param.u * u_r_rel,
        }
        A_r, b_r = solve_for_equilibrium(self.A, self.b, equilibrium, self.equilibrium_solution)

        self.A_r = A_r #.n() #matrix(R,A_r)
        self.b_r =  b_r #.n() #matrix(R,b_r)

        self.equilibrium = [ equilibrium['x1'],equilibrium['x2'],equilibrium['x3'],equilibrium['u']] 

        # u_r  = self.param.u*u_r_rel  
        # x_r1, x_r2, x_r3 = self.get_equilibrium(u_r)
        # print('equilibrium for nonlinear Threetank: ', x_r1, x_r2, x_r3)
        # self.equilibrium = [x_r1, x_r2, x_r3, u_r]
        # self.A_r, self.b_r = self.get_linearized_state_space(u_r, x_r1, x_r2, x_r3)


        A = matrix(R, Integer(3), Integer(4), [
            # 1. row
            self.A_r[0][0].n().simplest_rational() - x ,
            self.A_r[0][1].n().simplest_rational(),
            self.A_r[0][2].n().simplest_rational(),
            self.b_r[0][0].n().simplest_rational(),
            # 2. row
            self.A_r[1][0].n().simplest_rational() ,
            self.A_r[1][1].n().simplest_rational() - x,
            self.A_r[1][2].n().simplest_rational(),
            self.b_r[1][0].n().simplest_rational(),
            # 3. row
            self.A_r[2][0].n().simplest_rational(),
            self.A_r[2][1].n().simplest_rational(),
            self.A_r[2][2].n().simplest_rational() - x,
            self.b_r[2][0].n().simplest_rational(),
        ])
        return A, self.equilibrium
    
    def get_parameterized_ODEmatrix(self):
        A = matrix(R, Integer(3), Integer(4), [
            # 1. row
            self.A[0][0] - x ,
            self.A[0][1],
            self.A[0][2],
            self.b[0][0],
            # 2. row
            self.A[1][0] ,
            self.A[1][1] - x,
            self.A[1][2],
            self.b[1][0],
            # 3. row
            self.A[2][0],
            self.A[2][1],
            self.A[2][2] - x,
            self.b[2][0],
        ])
        return A
         
    
    def get_ODEfrom_spline(self, fkt: tuple):
        ode1 = lambda val: self.A_r[0][0] * fkt[0].derivative(val) + self.A_r[0][1] * fkt[1].derivative(val) + self.A_r[0][2] * fkt[2].derivative(val) + self.b_r[0][0] * fkt[3].derivative(val) - fkt[0].derivative(val,1) 
        ode2 = lambda val: self.A_r[1][0] * fkt[0].derivative(val) + self.A_r[1][1] * fkt[1].derivative(val) + self.A_r[1][2] * fkt[2].derivative(val) + self.b_r[1][0] * fkt[3].derivative(val) - fkt[1].derivative(val,1) 
        ode3 = lambda val: self.A_r[2][0] * fkt[0].derivative(val) + self.A_r[2][1] * fkt[1].derivative(val) + self.A_r[2][2] * fkt[2].derivative(val) + self.b_r[2][0] * fkt[3].derivative(val) - fkt[2].derivative(val,1) 

        return (ode1, ode2, ode3)
    
    def stateTransition(self, t, x, u=None, dt=None, model=None):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0

        if model is not None:
            with torch.no_grad():
                t_tensor = torch.tensor([t])
                outputs = model(torch.tensor([t]))
                reference = model.likelihood(outputs, train_data=model.train_inputs[0], current_data=t_tensor, mask=model.mask).mean.numpy() + model.equilibrium

            u_current = reference[0,3]
            control_idx = floor(abs(t/dt-0.000000001))
            u[control_idx] = u_current
        elif u is not None: #if isinstance(u, (list, tuple)):
            control_idx = floor(abs(t/dt-0.000000001))
            u_current = u[control_idx,0]

        

        dx0 = 1/self.param.A*(self.param._u* u_current-self.param.c13*sign(x[0]-x[2])*sqrt(2*self.param.g*abs(x[0]-x[2]))) #self.param.u is x[3]
        dx1 = 1/self.param.A*(self.param.c32*sign(x[2]-x[1])*sqrt(2*self.param.g*abs(x[2]-x[1]))-self.param.c2R*sqrt(2*self.param.g*abs(x[1])))
        dx2 = 1/self.param.A*(self.param.c13*sign(x[0]-x[2])*sqrt(2*self.param.g*abs(x[0]-x[2]))-self.param.c32*sign(x[2]-x[1])*sqrt(2*self.param.g*abs(x[2]-x[1])))
        #
        #dx3 = 0

        return [dx0, dx1, dx2]#, dx3
    
    def linear_stateTransition(self, t, x, u, dt):

        control_idx = floor(abs(t/dt-0.000000001))

        dx0 = self.A_r[0][0] * x[0] + self.A_r[0][1] * x[1] + self.A_r[0][2] * x[2] + self.b_r[0][0] * u[control_idx]
        dx1 = self.A_r[1][0] * x[0] + self.A_r[1][1] * x[1] + self.A_r[1][2] * x[2] + self.b_r[1][0] * u[control_idx]
        dx2 = self.A_r[2][0] * x[0] + self.A_r[2][1] * x[1] + self.A_r[2][2] * x[2] + self.b_r[2][0] * u[control_idx]
        #dx3 = 0

        return [dx0, dx1, dx2]#, dx3
        
