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
            u = 1e-4
            _u = 1
            #c13= 2.5e-5
            c12= 2.5e-5
            #c32= 2.5e-5
            c2R= 2e-5
            g= 9.81 
            A= 0.015

class Nonlinear_Watertank(ODE_System):
    
    x_min = [0, 0, 0]
    #x_max = [1, 1, 2e-4]
    x_max = [0.6, 0.6, 2e-4]
    
    def __init__(self):
        super().__init__(2,1)

        self.param = Parameter()

        x1, x2, u = var('x1, x2, u', domain='positive')#domain='real'

        f1 = symbolic_expression(1/self.param.A*(self.param._u * u-self.param.c12*sqrt(2*self.param.g*(x1-x2)))).function(x1,x2,u)
        f2 = symbolic_expression(1/self.param.A*(self.param.c12*sqrt(2*self.param.g*(x1-x2))-self.param.c2R*sqrt(2*self.param.g*(x2)))).function(x1,x2)
        

        self.state_var = [x1, x2]
        self.control_var = [u]
        self.system_eqations = [f1, f2] 

        solution, A, b = get_equilibrium_equations(self.system_eqations, self.state_var, self.control_var)

        self.equilibrium_solution = solution
        self.A = A
        self.b = b
    
    def get_ODEmatrix(self, u_r_rel:float):
        #TODO: the matrices A b and Ix could surely be merged in a more elegant way

        equilibrium ={
        'u': self.param.u * u_r_rel,
        }
        A_r, b_r = solve_for_equilibrium(self.A, self.b, equilibrium, self.equilibrium_solution)

        self.A_r = A_r #.n() #matrix(R,A_r)
        self.b_r =  b_r #.n() #matrix(R,b_r)

        self.equilibrium = [ equilibrium['x1'],equilibrium['x2'],equilibrium['u']] 

        A = matrix(R, Integer(2), Integer(3), [
            # 1. row
            self.A_r[0][0].n().simplest_rational() - x ,
            self.A_r[0][1].n().simplest_rational(),
            self.b_r[0][0].n().simplest_rational(),
            # 2. row
            self.A_r[1][0].n().simplest_rational() ,
            self.A_r[1][1].n().simplest_rational() - x,
            self.b_r[1][0].n().simplest_rational(),
        ])
        return A, self.equilibrium
    
    def stateTransition(self, t, x, u=None, dt=None, model=None):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0

        control_idx = floor(abs(t/dt-0.000000001))
        if model is not None:
            with torch.no_grad():
                t_tensor = torch.tensor([t])
                outputs = model(torch.tensor([t]))
                reference = model.likelihood(outputs, train_data=model.train_inputs[0], current_data=t_tensor, mask=model.mask).mean.numpy() #+ model.equilibrium

            u_current = reference[0,2]
            
            u[control_idx] = u_current
        elif u is not None: #if isinstance(u, (list, tuple)):
            u_current = u[control_idx,0]

        

        dx0 = 1/self.param.A*(self.param._u* u_current-self.param.c12*sign(x[0]-x[1])*sqrt(2*self.param.g*abs(x[0]-x[1]))) #self.param.u is x[3]
        dx1 = 1/self.param.A*(self.param.c12*sign(x[0]-x[1])*sqrt(2*self.param.g*abs(x[0]-x[1]))-self.param.c2R*sqrt(2*self.param.g*abs(x[1])))
        #
        #dx3 = 0

        return [dx0, dx1]#, dx3
    
    def linear_stateTransition(self, t, x, u, dt):

        control_idx = floor(abs(t/dt-0.000000001))

        dx0 = self.A_r[0][0] * x[0] + self.A_r[0][1] * x[1]  + self.b_r[0][0] * u[control_idx]
        dx1 = self.A_r[1][0] * x[0] + self.A_r[1][1] * x[1]  + self.b_r[1][0] * u[control_idx]

        return [dx0, dx1]#, dx3
        
