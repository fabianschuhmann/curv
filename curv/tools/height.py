import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
from ..core.fourier_core import Fourier_Series_Function
import os
from scipy.interpolate import RectBivariateSpline

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

def calc_height(out_dir,u,ndx,From=0,Until=None,Step=1,selection="name P"):
    if Until is None:
        Until=len(u.trajectory)
    ndx = read_ndx(ndx)
    box_size=u.trajectory[0].dimensions[:3]
    X,Y=get_XY(box_size)
    layer_group=u.atoms[[x-1 for x in ndx["Upper"]]]
    layer_group_2=u.atoms[[x-1 for x in ndx["Lower"]]]
    atoms_of_interest=u.select_atoms(selection)
    #atoms_of_interest=sel[np.in1d(sel.indices, layer_group.indices)]

    count = 0
    results={}
    for t,ts in enumerate(tqdm(u.trajectory[From:Until:Step])):
        count += 1
        #if count >= 10:
        #    break
        
        Nx, Ny = 2, 2
        fourier1=fourier_by_layer(layer_group,box_size)
        fourier2=fourier_by_layer(layer_group_2,box_size)
        fourier=Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
        fourier.Update_coff(fourier1.getAnm(),fourier2.getAnm()) #For the middle layer, update coefficients
        Z_fitted_1=np.array([fourier1.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
        Z_fitted_2=np.array([fourier2.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
        Z_fitted=(Z_fitted_1+Z_fitted_2)/2
        #Z_fitted=Z_fitted

        interp_func = RectBivariateSpline(X[0, :], Y[:, 0], Z_fitted)
        for atom in atoms_of_interest:
            pos=atom.position
            name=f"{atom.resname}_{atom.resid}"
            index=atom.index
            x_p, y_p, z_p = pos[0], pos[1], pos[2]  

            z_surf = interp_func(x_p, y_p)[0, 0]

            delta_z = np.abs(z_p - z_surf)

            try:
                results[index].append(delta_z)
            except KeyError:
                results[index]=[delta_z]

    for key,item in results.items():
        np.save(f"{out_dir}/height_{key}.npy",np.asarray(item)/10)


def height(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Calculate the relative height of a selection",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file")
    parser.add_argument('-n','--index',type=str,help="Specify the path to an index file containing the monolayers. To consider both monolayers, they need to be named 'Upper' and 'Lower'")
    parser.add_argument('-o','--out',type=str,help="Specify a path to a folder to which all calculated numpy arrays are saved")
    parser.add_argument('-F','--From',default=0,type=int,help="Discard all frames in the trajectory prior to the frame supplied here")
    parser.add_argument('-U','--Until',default=None,type=int,help="Discard all frames in the trajectory after to the frame supplied here")
    parser.add_argument('-l','--selection',type=str,help="Pass a selection for which the height should be calculated")
    parser.add_argument('-S','--Step',default=1,type=int,help="Traverse the trajectory with a step length supplied here")
    parser.add_argument('-c','--clear',default=False,action='store_true',help="Remove old numpy array in out directiory. NO WARNING IS GIVEN AND NO BACKUP IS MADE")
    
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if args.clear:
        for filename in os.listdir(args.out):
            if filename.endswith('.npy'):
                file_path = os.path.join(args.out, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    try:
        universe=mda.Universe(args.structure,args.trajectory)
        calc_height(out_dir=args.out,u=universe,ndx=args.index,From=args.From,Until=args.Until,Step=args.Step,selection=args.selection)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise