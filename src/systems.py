
import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import time
import torch
import matplotlib.pyplot as plt
from  lodegp import LODEGP 
from abc import ABC, abstractmethod

torch.set_default_tensor_type(torch.DoubleTensor)


class ODE_System(ABC):
    def __init__(self, dimension):
        self.dimension = dimension

    # @property
    # @abstractmethod
    # def dimension(self):
    #     pass

    @abstractmethod
    def get_ODEsolution(self, t_vec):
        pass

    @abstractmethod
    def get_ODEmatrix(self):
        pass

    @abstractmethod
    def get_ODEfrom_spline(self, fkt):
        pass

R = QQ['x']; (x,) = R._first_ngens(1)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# ODE_System 1
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

class System1(ODE_System):
    def __init__(self):
        super().__init__(3)

        class Parameter():
            pass

        self.param = Parameter()

    #ode1 = lambda val: fkt[0].derivative(val, 1) - fkt[1].derivative(val, 2) + fkt[1].derivative(val, 1) - fkt[1](val) + fkt[2].derivative(val, 1) - fkt[2](val)
    #ode2 = lambda val: 2*fkt[0](val) - fkt[0].derivative(val, 1) + fkt[1].derivative(val, 2) - fkt[1].derivative(val, 1) - fkt[1](val) - fkt[2].derivative(val, 

    def get_ODEsolution(self, t_vec):
        #TODO: would be nice to parameterize this too
        one = -0.25*torch.exp(t_vec) + 2*(torch.cos(t_vec) + torch.sin(t_vec)) 
        two = 4*torch.sin(t_vec) 
        three = -0.25*torch.exp(t_vec) + 2*(torch.cos(t_vec) - torch.sin(t_vec)) 

        return (one, two ,three)

    def get_ODEmatrix(self):
        A = matrix(R, Integer(2), Integer(3), [
            x,      -x**2+x-1,  x-2,
            2-x,    x**2-x-1,   -x
        ])
        return A

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Heating system 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#one = float(0.5)*torch.cos(float(0.5)*t_vec)*torch.exp(float(-0.1)*t_vec) + float(0.9)*torch.exp(float(-0.1)*t_vec)*torch.sin(float(0.5)*t_vec)
#two = torch.sin(float(0.5)*t_vec)*torch.exp(float(-0.1)*t_vec)
#three = float(1.9)*torch.cos(float(0.5)*t_vec)*torch.exp(float(-0.1)*t_vec) - float(16/25)*torch.exp(float(-0.1)*t_vec)*torch.sin(float(0.5)*t_vec)

# Heating system with parameters
#F = FunctionField(QQ, names=('a',)); (a,) = F._first_ngens(1)
#F = FunctionField(F, names=('b',)); (b,) = F._first_ngens(1)
#R = F['x']; (x,) = R._first_ngens(1)

#A = matrix(R, Integer(2), Integer(3), [x+a, -a, -1, -b, x+b, 0])
#self.model_parameters = torch.nn.ParameterDict({
#    "a":torch.nn.Parameter(torch.tensor(0.0)),
#    "b":torch.nn.Parameter(torch.tensor(0.0))
#})

#ode1 = lambda val: fkt[0].derivative(val, 1) + fkt[0](val)*a - fkt[1](val)*a - fkt[2](val)
#ode2 = lambda val: -fkt[0](val)*b + fkt[1].derivative(val, 1) + fkt[1](val)*b

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Bipendulum
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

class Bipendulum(ODE_System):
    def __init__(self):
        super().__init__(3)

        class Parameter():
            g = 9.81
            l1 = 1.0
            l2 = 2.0
        self.param = Parameter()

    def get_ODEsolution(self, t_vec):
        #TODO: would be nice to parameterize this too
        one   = -float(41)/float(100)*torch.sin(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(1))     - float(3)/float(5)*torch.cos(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(2))           + float(1)/float(5)*torch.sin(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(3))
        two   = float(81)/float(2000)*torch.sin(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(1))     - float(3)/float(10)*torch.cos(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(2))                   + float(1)/float(10)*torch.sin(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(3))
        three = -float(3321)/float(10000)*torch.sin(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(1)) + float(987)/float(500)*torch.cos(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(2)) - float(3929)/float(500)*torch.sin(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(3)) - float(36)/float(5)*torch.cos(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(4)) + float(12)/float(5)*torch.sin(float(3)*t_vec)/torch.pow((t_vec+int(1)), float(5))

        return (one, two ,three)

    def get_ODEmatrix(self):
        A = matrix(R, Integer(2), Integer(3), [
            x**2 + self.param.g/self.param.l1,    0,                                -1/self.param.l1, 
            0,                                    x**2+self.param.g/self.param.l2,  -1/self.param.l2
        ])
        return A

    def get_ODEfrom_spline(self, fkt: tuple):
        ode1 = lambda val: fkt[0].derivative(val, 2) + self.param.g/self.param.l1*fkt[0](val) - 1/self.param.l1*fkt[2](val)
        ode2 = lambda val: fkt[1].derivative(val, 2) + self.param.g/self.param.l2*fkt[1](val) - 1/self.param.l2*fkt[2](val)
        return (ode1, ode2)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Three tank
# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class ThreeTank(ODE_System):
    '''
    ThreeTank: (5 dimensional uncontrollable system)
    '''
    def __init__(self):
        super().__init__(5)

        class Parameter():
            pass

        self.param = Parameter()

    def get_ODEsolution(self, t_vec):
        #TODO: would be nice to parameterize this too
        one = float(1)*torch.exp(float(-0.5)*t_vec)
        two = float(1)*torch.exp(float(-0.25)*t_vec)
        three = float(1)*torch.exp(float(-0.25)*t_vec) - float(1)*torch.exp(float(-0.5)*t_vec)
        four = -float(0.5)*torch.exp(float(-0.5)*t_vec)
        five = - float(0.25)*torch.exp(float(-0.25)*t_vec) + float(0.5)*torch.exp(float(-0.5)*t_vec)

        return (one, two ,three, four, five)

    def get_ODEmatrix(self):
        A = matrix(R, Integer(3), Integer(5), [
            -x, 0, 0, 1, 0, 
            0, -x, 0, 1, 1, 
            0, 0, -x, 0, 1
        ])
        return A
    
    #A = matrix(R, Integer(3), Integer(5), [-x, 0, 0, 1, 0| 0, -x, 0, 1, 1| 0, 0, -x, 0, 1])
#ode1 = lambda val: -fkt[0].derivative(val, 1) + fkt[3](val)
#ode2 = lambda val: -fkt[1].derivative(val, 1) + fkt[3](val) + fkt[4](val)
#ode3 = lambda val: -fkt[2].derivative(val, 1) + fkt[4](val)
