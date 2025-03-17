import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
from ..core.fourier_core import Fourier_Series_Function
import os

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

def calc(out_dir,u,ndx,From=0,Until=None,Step=1,layer_string="Both"):
    if Until is None:
        Until=len(u.trajectory)
    ndx = read_ndx(ndx)
    box_size=u.trajectory[0].dimensions[:3]
    np.save(file=f"{out_dir}/boxsize.npy",arr=box_size)
    X,Y=get_XY(box_size)
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
        with mda.coordinates.XTC.XTCWriter(f"{out_dir}/fourier_curvature_fitting_{Layer}.xtc", n_atoms=100000) as writer:  # Adjust n_atoms as needed
            count = 0
            for t,ts in tqdm(enumerate(u.trajectory[From:Until:Step])):
                count += 1
                #if count >= 10:
                #    break
                
                if Layer=="Both":
                    Nx, Ny = 2, 2
                    fourier1=fourier_by_layer(layer_group,box_size)
                    fourier2=fourier_by_layer(layer_group_2,box_size)
                    fourier=Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
                    fourier.Update_coff(fourier1.getAnm(),fourier2.getAnm()) #For the middle layer, update coefficients
                    Z_fitted_1=np.array([fourier1.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
                    Z_fitted_2=np.array([fourier2.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
                    Z_fitted=np.abs(Z_fitted_1-Z_fitted_2)
                    Z_fitted_vmd=(Z_fitted_1+Z_fitted_2)/2

                else:
                    fourier=fourier_by_layer(layer_group,box_size)
                    Z_fitted = np.array([fourier.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
                    Z_fitted_vmd=Z_fitted[:]
                curvature = fourier.Curv(X, Y)
                coordinates = np.vstack([X.flatten(), Y.flatten(), Z_fitted_vmd.flatten()]).T
                Z_fitted=Z_fitted / 10 #Scale from \AA to nm

                np.save(f"{out_dir}/Z_fitted_{count}_{Layer}.npy",Z_fitted)

                
                curvature = curvature * 10 #Scale from \AA to nm

                np.save(f"{out_dir}/curvature_frame_{count}_{Layer}.npy", curvature)

                
                pseudo_universe = mda.Universe.empty(n_atoms=coordinates.shape[0], trajectory=True)
                pseudo_universe.atoms.positions = coordinates
                pseudo_universe.dimensions = ts.dimensions

                if t==0:
                    pseudo_universe.atoms.write(f"{out_dir}/pseudo_universe_{Layer}.gro")

                writer.write(pseudo_universe.atoms)

def calculate(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Calculate the curvature of a membrane",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file")
    parser.add_argument('-n','--index',type=str,help="Specify the path to an index file containing the monolayers. To consider both monolayers, they need to be named 'Upper' and 'Lower'")
    parser.add_argument('-o','--out',type=str,help="Specify a path to a folder to which all calculated numpy arrays are saved")
    parser.add_argument('-F','--From',default=0,type=int,help="Discard all frames in the trajectory prior to the frame supplied here")
    parser.add_argument('-U','--Until',default=None,type=int,help="Discard all frames in the trajectory after to the frame supplied here")
    parser.add_argument('-S','--Step',default=1,type=int,help="Traverse the trajectory with a step length supplied here")
    parser.add_argument('-l','--leaflet',default="Both",help="Choose which membrane leaflet to calculate. Default is Both")
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
        calc(out_dir=args.out,u=universe,ndx=args.index,From=args.From,Until=args.Until,Step=args.Step,layer_string=args.leaflet)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise