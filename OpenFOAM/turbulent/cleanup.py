#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:30:20 2021

@author: Vicky

Cleaning up the OpenFoam Simulation Data to make it ready for FNO 
"""

#  %%

from simvue import Simvue

INPUTS = []
OUTPUTS = ['Data/NS_turbulent.npz', 'Data/Re_Values.npy']

run = Simvue()

run.init(folder="/HPC-AI/turbulent", tags=['OpenFOAM', 'Turbulent', 'Processing', 'Cleanup'])

# %%
import os 
import numpy as np
from tqdm import tqdm
# %%
data_loc = os.getcwd() + '/Data'
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


#Only using Domain Data 
# %%

x_un = np.load(data_loc + '/' + time_list[-1] + '/NS_cyl_domain.npz')['x']
y_un = np.load(data_loc + '/' + time_list[-1] + '/NS_cyl_domain.npz')['y']

# %%
'''

u_list = []
v_list = []
p_list = []

ngridx = 101
ngridy = 101

xi = np.linspace(-5.0, 5.0, ngridx)
yi = np.linspace(-5.0, 5.0, ngridy)
Xi, Yi = np.meshgrid(xi, yi)

from scipy.interpolate import griddata

for sim in tqdm(time_list[:3]):
    domain_data = np.load(data_loc + '/' + sim + '/NS_cyl_Domain.npz')
    bound_data = np.load(data_loc + '/' + sim + '/NS_cyl_boundary_outputs.npz')
    
    Ui = []
    Vi = []
    Pi = []
    for it in range(50):
        u = domain_data['U'][it][0]
        ui = griddata((x_un, y_un), u, (xi[None, :], yi[:, None]), method='linear')[1:-1, 1:-1]
        Ui.append(ui)
        v = domain_data['U'][it][0]
        vi = griddata((x_un, y_un), v, (xi[None, :], yi[:, None]), method='linear')[1:-1, 1:-1]
        Vi.append(vi)
        p = domain_data['P'][it]
        pi = griddata((x_un, y_un), p, (xi[None, :], yi[:, None]), method='linear')[1:-1, 1:-1]
        Pi.append(pi)
    
    u_list.append(np.asarray(Ui))
    v_list.append(np.asarray(Vi))
    p_list.append(np.asarray(Pi))


u_list = np.asarray(u_list)
v_list = np.asarray(v_list)
p_list = np.asarray(p_list)

'''
# %%
#Including Boundary Information as well. 

bound_xy = np.load(data_loc + '/' + time_list[-1] + '/NS_cyl_boundary_inputs.npz')
x_un_b = []
y_un_b = []
for ii, loc in enumerate(bound_xy.files):
    if ii%2 == 0 :
        x_un_b.append(bound_xy[loc].ravel())
    if ii%2 == 1 :
        y_un_b.append(bound_xy[loc].ravel())

x_un_b = np.concatenate(x_un_b).ravel()
y_un_b = np.concatenate(y_un_b).ravel()

# %%

x_inlet = bound_xy['x_inlet']
y_inlet = bound_xy['y_inlet']
x_outlet = bound_xy['x_outlet']
y_outlet = bound_xy['y_outlet']
x_ymin = bound_xy['x_ymin']
y_ymin = bound_xy['y_ymin']
x_ymax = bound_xy['x_ymax']
y_ymax = bound_xy['y_ymax']
x_cyl = bound_xy['x_walls']
y_cyl = bound_xy['y_walls']

X = np.append([x_un], [x_un_b])
Y = np.append([y_un], [y_un_b])
# %%

u_list = []
v_list = []
p_list = []

ngridx = 101
ngridy = 101

xi = np.linspace(-5.0, 5.0, ngridx)
yi = np.linspace(-5.0, 5.0, ngridy)
Xi, Yi = np.meshgrid(xi, yi)

from scipy.interpolate import griddata

for sim in tqdm(time_list):
    domain_data = np.load(data_loc + '/' + sim + '/NS_cyl_domain.npz')
    bound_data = np.load(data_loc + '/' + sim + '/NS_cyl_boundary_outputs.npz')
    
    Ui = []
    Vi = []
    Pi = []
    for it in range(20):
        run.log_metrics({'Iterations': it})
        
        u = domain_data['U'][it][0]
        u_inlet = bound_data['U_inlet'][it][0]
        u_outlet = bound_data['U_outlet'][it][0]
        u_ymin = bound_data['U_ymin'][it][0]
        u_ymax = bound_data['U_ymax'][it][0]
        u_cylinder = np.ones(len(x_cyl))*bound_data['U_cylinder'][it][0]
        U = np.vstack((u[:, np.newaxis], u_inlet[:,np.newaxis], u_outlet[:,np.newaxis], u_ymin[:,np.newaxis], u_ymax[:,np.newaxis], u_cylinder[:,np.newaxis]))
        U = U.flatten()
        ui = griddata((X, Y), U, (xi[None, :], yi[:, None]), method='linear')[1:-1, 1:-1]
        Ui.append(ui)
        
        v = domain_data['U'][it][1]
        v_inlet = bound_data['U_inlet'][it][1]
        v_outlet = bound_data['U_outlet'][it][1]
        v_ymin = bound_data['U_ymin'][it][1]
        v_ymax = bound_data['U_ymax'][it][1]
        v_cylinder = np.ones(len(x_cyl))*bound_data['U_cylinder'][it][1]
        V = np.vstack((v[:,np.newaxis], v_inlet[:,np.newaxis], v_outlet[:,np.newaxis], v_ymin[:,np.newaxis], v_ymax[:,np.newaxis], v_cylinder[:,np.newaxis]))
        V = V.flatten()
        vi = griddata((X, Y), V, (xi[None, :], yi[:, None]), method='linear')[1:-1, 1:-1]
        Vi.append(vi)
        
        p = domain_data['P'][it]
        p_inlet = np.ones(len(x_cyl))*bound_data['P_inlet'][it]
        p_outlet =np.ones(len(x_cyl))*bound_data['P_outlet'][it]
        p_ymin = bound_data['P_ymin'][it]
        p_ymax = bound_data['P_ymax'][it]
        p_cylinder = bound_data['P_cylinder'][it]
        P = np.vstack((p[:,np.newaxis], p_inlet[:,np.newaxis], p_outlet[:,np.newaxis], p_ymin[:,np.newaxis], p_ymax[:,np.newaxis], p_cylinder[:,np.newaxis]))
        P = P.flatten()
        pi = griddata((X, Y), P, (xi[None, :], yi[:, None]), method='linear')[1:-1, 1:-1]
        Pi.append(pi)
        
    
    u_list.append(np.asarray(Ui))
    v_list.append(np.asarray(Vi))
    p_list.append(np.asarray(Pi))


u_list = np.asarray(u_list)
v_list = np.asarray(v_list)
p_list = np.asarray(p_list)

# %%
t = np.arange(0, 400+20, 20)
np.savez(data_loc + '/NS_turbulent', U=u_list, V=v_list, P = p_list, x=xi[1:-1], y=yi[1:-1],t=t)


# %%
# data=np.load(data_loc[:-5]+'/NS_Cyl.npz')

# %%

# Save input files
for input_file in INPUTS:
    if os.path.isfile(input_file):
        run.save(input_file, 'input', 'text/plain')
    elif os.path.isdir(input_file):
        run.save_directory(input_file, 'input', 'text/plain', preserve_path=True)
    else:
        print('ERROR: input file %s does not exist' % input_file)
        
            
            
# Save output files
for output_file in OUTPUTS:
    if os.path.isfile(output_file):
        run.save(output_file, 'output')
    elif os.path.isdir(output_file):
        run.save_directory(output_file, 'output', 'text/plain', preserve_path=True)   
    else:
        print('ERROR: output file %s does not exist' % output_file)
    
run.close()