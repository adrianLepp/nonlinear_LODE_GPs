from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
import torch
import numpy as np
from nonlinear_LODE_GPs.systems import ODE_System

R = QQ['x']; (x,) = R._first_ngens(1)

class Inverted_Pendulum(ODE_System):
    def __init__(self):
        super().__init__(3)

        class Parameter():
            M = 0.5
            m = 0.2
            b = 0.1
            I = 0.006
            l = 0.3
            g = 9.81

        self.param = Parameter()

    def get_ODEmatrix2(self):
        A = matrix(R, Integer(2), Integer(3), [
            # 1. row
            -(self.param.I + self.param.m * self.param.l**2 )*self.param.b*  x - (self.param.I * (self.param.M + self.param.m ) + self.param.M * self.param.m * self.param.l**2)* x**2 ,
            (self.param.m**2 * self.param.g * self.param.l),
            (self.param.I + self.param.m * self.param.l**2),
            # 2. row
            (-self.param.m * self.param.l * self.param.b) * x,
            (self.param.m * self.param.g * self.param.l * (self.param.M + self.param.m)) - (self.param.I * (self.param.M + self.param.m) + self.param.M * self.param.m * self.param.l**2) * x**2,
            (self.param.m * self.param.l)
        ])
        return A
    
    def get_ODEmatrix(self):
        A = matrix(R, Integer(2), Integer(3), [
            # 1. row
            (- self.param.b / self.param.M) * x - x**2 ,
            (- self.param.m * self.param.g / self.param.M),
            + 1 / self.param.M,
            # 2. row
            (self.param.b / (self.param.M * self.param.l)) * x,
            ((self.param.M + self.param.m) * self.param.g / (self.param.M * self.param.l)) - x**2,
            - 1 / (self.param.M * self.param.l)
        ])
        return A

    def get_ODEfrom_spline2(self, fkt: tuple):
        ode1 = lambda val: (-(self.param.I + self.param.m * self.param.l**2 )*self.param.b) * fkt[0].derivative(val, 1) - (self.param.I * (self.param.M + self.param.m ) + self.param.M * self.param.m * self.param.l**2)* fkt[0].derivative(val, 2) + (self.param.m**2 * self.param.g * self.param.l) * fkt[1].derivative(val) + (self.param.I + self.param.m * self.param.l**2) * fkt[2].derivative(val)
        ode2 = lambda val: (-self.param.m * self.param.l * self.param.b) * fkt[0].derivative(val, 1) + (self.param.m * self.param.g * self.param.l * (self.param.M + self.param.m)) * fkt[1].derivative(val) - (self.param.I * (self.param.M + self.param.m) + self.param.M * self.param.m * self.param.l**2) * fkt[1].derivative(val, 2) + (self.param.m * self.param.l) * fkt[2].derivative(val)

        return (ode1, ode2)
    
    def get_ODEfrom_spline(self, fkt: tuple):
        ode1 = lambda val: - self.param.b *  fkt[0].derivative(val, 1) - self.param.M * fkt[0].derivative(val, 2) - self.param.m * self.param.g * fkt[1].derivative(val) + fkt[2].derivative(val)
        ode2 = lambda val: self.param.b * fkt[0].derivative(val, 1) + self.param.g * (self.param.M + self.param.m) * fkt[1].derivative(val) - self.param.l * self.param.M * fkt[1].derivative(val, 2) - fkt[2].derivative(val)

        return (ode1, ode2)
    
    def stateTransition2(self, t, x):
        numerator = (self.param.I * (self.param.M + self.param.m ) + self.param.M * self.param.m * self.param.l**2)
        dx0 = x[1] 
        dx1 = (- (self.param.I + self.param.m * self.param.l**2 ) *self.param.b) / numerator * x[1] + (self.param.m**2 * self.param.g * self.param.l**2) / numerator * x[2] + (self.param.I + self.param.m * self.param.l**2) / numerator * x[4]
        dx2 = x[3]
        dx3 = (-self.param.m * self.param.l * self.param.b) / numerator * x[1] + (self.param.m * self.param.g * self.param.l * (self.param.M + self.param.m)) / numerator * x[2] +(self.param.m * self.param.l) / numerator * x[4]
        dx4 = 0

        return [dx0, dx1, dx2, dx3, dx4]
    
    def stateTransition(self, t, x):
        #
        dx0 = x[1] 
        #
        pt1a = - self.param.m * self.param.g * np.sin(x[2]) * np.cos(x[2])
        pt1b = - self.param.b * x[1]
        pt1c = self.param.m * self.param.l * np.sin(x[2]) * x[3]**2  
        pt1d = x[4]
        numerator1 = self.param.M + self.param.m * np.sin(x[2])**2
        dx1 =   (pt1a + pt1b + pt1c + pt1d)/(numerator1)
        #
        dx2 = x[3]
        #
        pt3a = (self.param.M + self.param.m) * self.param.g * np.sin(x[2])
        pt3b = self.param.b * np.cos(x[1]) * x[1]
        pt3c = - self.param.m * self.param.l * np.sin(x[2]) * np.cos(x[2]) * x[3]**2  
        pt3d = - np.cos(x[2]) * x[4]
        numerator3 = self.param.l * (self.param.M + self.param.m * np.sin(x[2])**2 )
        dx3 =   (pt3a + pt3b + pt3c + pt3d)/(numerator3)
        #
        dx4 = 0

        return [dx0, dx1, dx2, dx3, dx4]
    
    def stateTransition3(self, t, x):
        dx0 = x[1]
        dx1 = x[1] * (- self.param.b / self.param.M) + x[2] * (- self.param.m * self.param.g / self.param.M) + x[4] / self.param.M
        dx2 = x[3]
        dx3 = x[1] * (self.param.b / (self.param.M * self.param.l)) + x[2] * ((self.param.M + self.param.m) * self.param.g / (self.param.M * self.param.l)) + x[4] / ( - self.param.M * self.param.l)
        dx4 = 0

        return [dx0, dx1, dx2, dx3, dx4]