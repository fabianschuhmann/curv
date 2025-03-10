import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
from ..core.fourier_core import Fourier_Series_Function
import os
import concurrent.futures
import functools

logger = logging.getLogger(__name__)

def get_XY(box_size):
    x = np.linspace(0, box_size[0], 100)
    y = np.linspace(0, box_size[1], 100)
    X, Y = np.meshgrid(x, y)
    return X,Y

def read_ndx(filename):
    groups = {}
    with open(filename) as f:
        group_name = None
        for line in f:
            line = line[:line.find(";")].strip()
            if line.startswith('['):  
                group_name = line[1:-1].strip()
                groups[group_name] = []
            elif group_name is not None:
                groups[group_name].extend(map(int, line.split()))
    return groups

def fourier_by_layer(layer_group,box_size,Nx=2,Ny=2):
    Lx = box_size[0]
    Ly = box_size[1]
    data_3m=layer_group.positions.T
    fourier = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    fourier.Fit(data_3m)

    return fourier

def parallal_frames(u,ndx,box_size,layer_string):
    X,Y=get_XY(box_size)
    result=[]
    if layer_string.lower()!="Both".lower():
        LayerList=[layer_string]
    else:
        LayerList=["Upper","Lower","Both"]
    for Layer in LayerList:
        if Layer=="Both":
            layer_group=u.atoms[[x-1 for x in ndx["Upper"]]]
            layer_group_2=u.atoms[[x-1 for x in ndx["Lower"]]]
        else:
            layer_group=u.atoms[[x-1 for x in ndx[Layer]]]
        if Layer=="Both":
            Nx, Ny = 2, 2
            fourier1=fourier_by_layer(layer_group,box_size)
            fourier2=fourier_by_layer(layer_group_2,box_size)
            fourier=Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
            fourier.Update_coff(fourier1.getAnm(),fourier2.getAnm()) #For the middle layer, update coefficients
            Z_fitted_1=np.array([fourier1.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
            Z_fitted_2=np.array([fourier2.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
            Z_fitted=np.abs(Z_fitted_1-Z_fitted_2)
        else:
            fourier=fourier_by_layer(layer_group,box_size)
            Z_fitted = np.array([fourier.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
        curvature = fourier.Curv(X, Y)
        coordinates = np.vstack([X.flatten(), Y.flatten(), Z_fitted.flatten()]).T
        Z_fitted=Z_fitted / 10 #Scale from \AA to nm
        
        curvature = curvature * 10 #Scale from \AA to nm

        result.append((Z_fitted,curvature,Layer))
    return result

def calc_p(out_dir,u,ndx,From=0,Until=None,Step=1,layer_string="Both",worker=2):
    if Until is None:
        Until=len(u.trajectory)
    ndx = read_ndx(ndx)
    box_size=u.trajectory[0].dimensions[:3]
    np.save(file=f"{out_dir}/boxsize.npy",arr=box_size)

    partial_func = functools.partial(parallal_frames, ndx=ndx, box_size=box_size, layer_string=layer_string)
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        results=list(executor.map(partial_func,u.trajectory[From:Until:Step]))
    
    #one might consider giving the count list to the parallel and have it save directly, is then hard drive write speed the bottleneck?
    for count,tupels in enumerate(results):
        for result in tupels:
            np.save(f"{out_dir}/Z_fitted_{count+1}_{result[2]}.npy",result[0])
            np.save(f"{out_dir}/curvature_frame_{count+1}_{result[2]}.npy", result[1])
