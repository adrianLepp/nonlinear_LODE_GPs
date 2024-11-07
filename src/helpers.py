import gpytorch 
from gpytorch.kernels.kernel import Kernel
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
import pprint
import time
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

def calc_finite_differences(sample, point_step_size, skip=False, number_of_samples=0):
    """
    param skip: Decides whether to skip every second value of the sample.
                Useful for cases where original samples aren't equidistant
    """
    if sample.ndim == 2:
        NUM_CHANNELS = sample.shape[1]
    else:
        NUM_CHANNELS = 1
    if number_of_samples == 0:
        number_of_samples = sample.shape[0]

    gradients_list = list()
    if skip:
        step = 2
    for index in range(0, step*number_of_samples, step):
        gradients_list.append(list((-sample[index] + sample[index+1])/point_step_size))
    return gradients_list

def saveDataToCsv(folder:str, simName:str, data:dict, overwrite:bool=False):
    fileName = folder + simName +  '.csv'

    if os.path.exists(fileName) and not overwrite:
        raise FileExistsError(f"The file {fileName} already exists.")
    
    df = pd.DataFrame(data)
    df.to_csv(fileName, index=False)


def collectMetaInformation(id:int,model_name:str, system_name:str, parameters, rms):
    info = {}
    info['date'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #info['gp_param'] = dict(parameters)
    info['model'] = model_name
    info['system'] = system_name
    info['rms'] = rms
    info['id'] = id

    return info

def saveSettingsToJson(folder:str, simName:str, settings:dict, overwrite:bool=False):
    fileName = folder + simName + '_settings.json'

    if os.path.exists(fileName) and not overwrite:
        raise FileExistsError(f"The file {fileName} already exists.")

    with open(folder + simName + '.json',"w") as f:
        json.dump(settings, f)