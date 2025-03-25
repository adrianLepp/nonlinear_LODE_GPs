from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
import torch
import numpy as np
from nonlinear_LODE_GPs.systems import ODE_System
from nonlinear_LODE_GPs.systems.linearize import solve_for_equilibrium, get_equilibrium_equations


# R = QQ['x, x1, x2, x3, u']
# (x, x1, x2, x3, u) = R._first_ngens(5)

R = QQ['x']; (x,) = R._first_ngens(1)

class Parameter():
    def __init__(self, a0:float,a1:float, v:float):
        self.a0 = a0
        self.a1 = a1
        self.v = v
        self.g = 9.81
        self.m = 1
        self.l  =1.5
        self.d = 0.5
    
            

class Inverted_Pendulum(ODE_System):
    def __init__(self, a0:float,a1:float, v:float):
        super().__init__(2,1)


        self.param = Parameter(a0,a1,v)

        x1, x2, u = var('x1, x2, u', domain='positive')#domain='real'

        f1 = symbolic_expression(x2).function(x2)
        f2 = symbolic_expression(-self.param.g/self.param.l*cos(x1) - self.param.d*x2 - 1/self.param.m * u ).function(x1,x2,u)
        

        self.state_var = [x1, x2]
        self.control_var = [u]
        self.system_eqations = [f1, f2] 

        self.A = matrix(R, Integer(2), Integer(2), [
            0, 1,
            - self.param.a0, - self.param.a1
        ])
        self.b = matrix(R, Integer(2), Integer(1), [
            0,
            self.param.v
        ])

    def get_ODEmatrix(self):
        A = self.A
        b = self.b

        V = matrix(R, Integer(2), Integer(3), [
            # 1. row
            A[0][0].n().simplest_rational() - x ,
            A[0][1].n().simplest_rational(),
            b[0][0].n().simplest_rational(),
            # 2. row
            A[1][0].n().simplest_rational() ,
            A[1][1].n().simplest_rational() - x,
            b[1][0].n().simplest_rational(),
        ])
        return V
     
    def stateTransition(self, t, x, u=None, dt=None, model=None):
        u_current = self.get_control_from_list(t, dt, u)

        dx0 = x[1]
        dx1 = -self.param.g/self.param.l*np.cos(x[0]) - self.param.d*x[1] - 1/self.param.m * u_current
        return [dx0, dx1]
    
    # def stateTransition_2(self, t, x, y_ref_control=None, dt=None, direct_control=False, u=None): #FIXME how do you call me
    #     if y_ref_control is None:
    #         y_ref = 0
    #     else:
    #         y_ref = self.get_control_from_list(t, dt, y_ref_control)

    #     if direct_control:
    #         u_current = + self.polynom(x) + self.param.v * y_ref
    #     else:
    #         u_current = self.get_control_from_latent(y_ref,x).squeeze()

    #     if u is not None:
    #         idx = floor(abs(t/dt-0.000000001))
    #         print(idx)
    #         u[floor(abs(t/dt-0.000000001))] = u_current #FIXME: this is not correct, since multiple values are written to the same index (due to Runge Kutta)

    #     dx0 = x[1]
    #     dx1 = -self.param.g/self.param.l*np.cos(x[0]) - self.param.d*x[1] - 1/self.param.m * u_current
    #     return [dx0, dx1]

    def stateTransition_2(self, t, x, dt=None, controller=None, u=None, y_ref=None): #FIXME how do you call me
        if controller is not None:
            if y_ref is None:
                y_ref_current = 0
            else:
                y_ref_current = self.get_control_from_list(t, dt,  y_ref)
            u_current = controller(x, y_ref_current)
            idx = floor(abs(t/dt-0.000000001))
            u[idx] = u_current

            #u_current = self.get_control_from_latent(y_ref,x).squeeze()
        
        elif u is not None:
            u_current = self.get_control_from_list(t, dt, u)

        else:
            raise ValueError('No control input given')

        dx0 = x[1]
        dx1 = -self.param.g/self.param.l*np.cos(x[0]) - self.param.d*x[1] + 1/self.param.m * u_current
        return [dx0, dx1]
    
    def get_latent_control(self, u:float, x:np.ndarray):
        y_ref = (self.alpha(x) + self.polynom(x) + u*self.beta(x,u))  / self.param.v 
        return y_ref
    
    def get_control_from_latent(self, y_ref:float, x:np.ndarray):
        u = self.param.v / self.beta(x, 0) * y_ref -  (self.alpha(x) + self.polynom(x)) / self.beta(x,0) #FIXME how to chooose u in beta
        return u
    
    def get_control_from_list(self, t:float, dt:float, u_list:np.ndarray):
        control_idx = floor(abs(t/dt-0.000000001))
        u_current = u_list[control_idx].squeeze()
        return u_current

    def linear_stateTransition(self, t, x:np.ndarray, u, dt):
        u_current = self.get_control_from_list(t, dt, u)

        y_ref = self.get_latent_control(u_current,x)

        dx0 = self.A[0][0] * x[0] + self.A[0][1] * x[1]  + self.b[0][0] * y_ref
        dx1 = self.A[1][0] * x[0] + self.A[1][1] * x[1]  + self.b[1][0] * y_ref

        return [dx0, dx1]
    
    def polynom(self, x:np.ndarray):
        return self.param.a0 * x[0] + self.param.a1 * x[1]
     
    def alpha(self, x:np.ndarray):
        return -self.param.g/self.param.l * np.cos(x[0]) - self.param.d * x[1]

    def beta(self, x, u=None):
        return 1/self.param.m
    
    def rad_to_deg(self, x):
        return x * [180 / np.pi, 180 / np.pi, 1]
