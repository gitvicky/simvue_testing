#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:35:44 2022

@author: vgopakum

Generating the Dataset for FNO tests by varying Reynolds number (Kinematic Viscosity) to be between 40 and 400. 
Data Extraction is performed using fluidfloam package
"""
# %%
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs

from fluidfoam.readof import readmesh
from fluidfoam.readof import readvector
from fluidfoam.readof import readscalar
import subprocess

# %%
data_loc = os.getcwd()
subprocess.call(['rm', '-r', os.getcwd()+'/Data'])
os.mkdir(data_loc + '/Data/')

# %%
num_sims = 5
lb = np.asarray([0.005]) # length, width
ub = np.asarray([0.01]) # length, width
Re_list = np.round(lb + (ub - lb) * lhs(1, num_sims), 6)

np.save(data_loc + '/Data/Re_Values', Re_list)

subprocess.call(["cp", data_loc + '/transportProperties', data_loc + '/constant/transportProperties'])

current_value = 0.01
prev_Re = current_value
for sim in range(num_sims):
    
    Re = Re_list[sim][0]
    print('Prev Re ' + str(prev_Re))
    print('Re '+ str(Re))

    # subprocess.call('./Allclean')

    subprocess.call(["sed", "-i", "-e",  's/'+str(prev_Re)+'/'+str(Re)+'/g', os.getcwd() + "/constant/transportProperties"])

    # subprocess.call(['blockMesh'])
    # subprocess.call('./Allrun')
    subprocess.call(['./run_solver.sh', str(sim)])

    # %%
    os.mkdir(os.getcwd()+'/Data/'+str(sim+1))
    save_loc = os.getcwd()+'/Data/'+str(sim+1)
    subprocess.call(["cp", data_loc + '/constant/transportProperties',save_loc])

    # %%
    # Creating a list of all the time instance directories
    dir_list = os.listdir(data_loc)
    time_list = []

    for directory in dir_list:
        try:
            float(directory)
            time_list.append(directory)
        except:
            pass
    time_list.sort(key=float)
    time_list = np.array(time_list)
    # %%
    x, y, z = readmesh(data_loc, structured=False)


    # %%
    U_series = []
    p_series = []
    phi_series = []

    for timename in time_list[1:]:
        U = readvector(data_loc, timename, 'U', structured=False)
        p = readscalar(data_loc, timename, 'p', structured=False)
        phi = readscalar(data_loc, timename, 'phi', structured=False)

        U_series.append(U)
        p_series.append(p)
        phi_series.append(phi)

    # %%
    U = np.asarray(U_series)
    P = np.asarray(p_series)
    Phi = np.asarray(phi_series)

    t = np.arange(0, 400+20, 20)
    np.savez(save_loc +'/NS_cyl_domain', x=x, y=y, t=t, U=U, P=P, Phi=Phi)


    # %%
    # Boundary Information

    x_inlet, y_inlet, z_inlet = readmesh(data_loc, structured=True, boundary="in")
    x_outlet, y_outlet, z_outlet = readmesh(data_loc, structured=True, boundary="out")
    x_ymin, y_ymin, z_ymin = readmesh(data_loc, structured=True, boundary="sym2")
    x_ymax, y_ymax, z_ymax = readmesh(data_loc, structured=True, boundary="sym1")
    x_cylinder, y_cylinder, z_cylinder = readmesh(data_loc, structured=False, boundary="cylinder")

    np.savez(save_loc + '/NS_cyl_boundary_inputs', x_inlet=x_inlet, y_inlet=y_inlet, x_outlet=x_outlet, y_outlet=y_outlet,
             x_ymin=x_ymin, y_ymin=y_ymin, x_ymax=x_ymax, y_ymax=y_ymax, x_walls=x_cylinder, y_walls=y_cylinder)

    # %%
    boundaries = ['in', 'out', 'sym2', 'sym1', 'cylinder']
    bound_names = ['inlet', 'outlet', 'ymin', 'ymax', 'cylinder']
    bound_values = {}
    for it, bound in enumerate(boundaries):
        U_series = []
        p_series = []
        phi_series = []

        for timename in time_list[1:]:
            U = readvector(data_loc, timename, 'U', structured=False, boundary=bound)
            p = readscalar(data_loc, timename, 'p', structured=False, boundary=bound)
            phi = readscalar(data_loc, timename, 'phi', structured=False, boundary=bound)

            U_series.append(U)
            p_series.append(p)
            phi_series.append(phi)

        U = np.asarray(U_series)
        P = np.asarray(p_series)
        Phi = np.asarray(phi_series)

        globals()[bound_names[it] + '_U'] = U
        globals()[bound_names[it] + '_P'] = P
        globals()[bound_names[it] + '_Phi'] = Phi

    # %%
    np.savez(save_loc+'/NS_cyl_boundary_outputs', U_inlet=inlet_U, P_inlet=inlet_P, Phi_inlet=inlet_Phi,
             U_outlet=outlet_U, P_outlet=outlet_P, Phi_outlet=outlet_Phi,
             U_ymin=ymin_U, P_ymin=ymin_P, Phi_ymin=ymin_Phi,
             U_ymax=ymax_U, P_ymax=ymax_P, Phi_ymax=ymax_Phi,
             U_cylinder=cylinder_U, P_cylinder=cylinder_P, Phi_cylinder=cylinder_Phi)

    prev_Re = Re

