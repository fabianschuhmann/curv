import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
from ..core.fourier_core import Fourier_Series_Function
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_XY(box_size):
    x = np.linspace(0, box_size[0], 100)
    y = np.linspace(0, box_size[1], 100)
    X, Y = np.meshgrid(x, y)
    return X,Y

def draw(Dir,layer1="Upper",layer2="Lower",layer3="Both",minmax=None,filename="", rotate=False):
    # Plots
    fontsize=24
    box_size=np.load(Dir+"boxsize.npy")
    X,Y = get_XY(box_size)
    center_x = X/2
    center_y = Y/2

    with open('radius_threshold.txt', 'r') as f:
        radius_threshold = int(float(f.read()))

    distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = distance_from_center <= radius_threshold

    # Upper layer
    prefix = "curvature_frame_" if rotate == False else "curvature_rotation_"
    curvature_data1=[]
    for file_path in glob.glob(Dir+prefix+f"*_{layer2}.npy"):
        curvature_data1.append(np.load(file_path))
    curvature_data1 = np.asarray(curvature_data1)
    curvature_data1=np.mean(curvature_data1,axis=0)

    # Lower layer 
    curvature_data2=[]
    for file_path in glob.glob(Dir+prefix+f"*_{layer1}.npy"):
        curvature_data2.append(np.load(file_path))
    curvature_data2 = np.asarray(curvature_data2)
    curvature_data2=np.mean(curvature_data2,axis=0)

    # Middle layer
    curvature_data3=[]
    for file_path in glob.glob(Dir+prefix+f"*_{layer3}.npy"):
        curvature_data3.append(np.load(file_path))
    curvature_data3 = np.asarray(curvature_data3)
    curvature_data3=np.mean(curvature_data3,axis=0)

    #if(rotate == True):
    #    curvature_data1 = np.ma.masked_where(~mask , curvature_data1)
    #    curvature_data2 = np.ma.masked_where(~mask , curvature_data2)
    #    curvature_data3 = np.ma.masked_where(~mask , curvature_data3)

    catted=[curvature_data1,curvature_data2,curvature_data3]

    if minmax is None:
        Minimum, Maximum =np.min([np.min(x) for x in catted]),np.max([np.max(x) for x in catted])
    else:
        Minimum, Maximum = minmax[0],minmax[1]
        print(f"A custom minimum and maximum has been set to {Minimum} and {Maximum}.")
        bmin, bmax =np.min([np.min(x) for x in catted]),np.max([np.max(x) for x in catted])
        print(f"Automatically, the values would have been assigned to {np.round(bmin,3)} and {np.round(bmax,3)}")


    # Thickness
    prefix = "Z_fitted_" if rotate == False else "Z_fitted_rotation_"
    Z_fitted=[]
    for file_path in glob.glob(Dir+prefix+f"*_{layer3}.npy"):
        Z_fitted.append(np.load(file_path))
    Z_fitted = np.asarray(Z_fitted)
    Z_fitted=np.mean(Z_fitted,axis=0)

    #if(rotate == True):
    #    curvature_data3 = np.ma.masked_where(~mask , curvature_data3)


    fig, axes = plt.subplots(2, 2, figsize=(24,28))  # Create 1 row, 2 columns of subplots
    axes=axes.flatten()
    for ax in axes:
        ax.set_aspect(box_size[0]/box_size[1])
        ax.set_xticks([0,box_size[0]])
        ax.set_yticks([0,box_size[1]])
        ax.set_xticklabels(['0', 'L$_x$'], fontsize=fontsize)  # Increase font size
        ax.set_yticklabels(['0', 'L$_y$'], fontsize=fontsize)


    # First subplot: Fourier Approximation
    contour1 = axes[0].contourf(X, Y, Z_fitted, cmap="viridis")
    axes[0].set_title("Fourier Approximation",fontsize=fontsize)
    #axes[0].set_xlabel("X [nm]")
    #axes[0].set_ylabel("Y [nm]")
    levels=np.linspace(Minimum,Maximum,20)
    # Second subplot: Curvature
    norm = mcolors.Normalize(vmin=Minimum, vmax=Maximum)
    contour4 = axes[3].contourf(X, Y, curvature_data3, cmap="plasma",norm=norm,extend="both",levels=levels)
    axes[3].set_title(f"Curvature {layer3}",fontsize=fontsize)
    #axes[3].set_xlabel("X [nm]")
    #axes[3].set_ylabel("Y [nm]")

    # Second subplot: Curvature
    contour2 = axes[1].contourf(X, Y, curvature_data1, cmap="plasma",norm=norm,extend="neither",levels=levels)
    axes[1].set_title(f"Curvature {layer1}",fontsize=fontsize)
    #axes[1].set_xlabel("X [nm]")
    #axes[1].set_ylabel("Y [nm]")

    # Second subplot: Curvature
    contour3 = axes[2].contourf(X, Y, curvature_data2, cmap="plasma",norm=norm,extend="both",levels=levels)
    axes[2].set_title(f"Curvature {layer2}",fontsize=fontsize)
    #axes[2].set_xlabel("X [nm]")
    #axes[2].set_ylabel("Y [nm]")


    cbar_ax = fig.add_axes([0.08, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(contour1, cax=cbar_ax)
    cbar_ax.set_ylabel("Thickness ($nm$)",fontsize=fontsize)
    cbar_ax.yaxis.set_ticks_position('left')
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.tick_params(labelsize=fontsize)

    cbar_ax2 = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(contour2, cax=cbar_ax2)
    cbar_ax2.tick_params(labelsize=fontsize)
    cbar_ax2.set_ylabel("Curvature ($nm^{-1}$)",fontsize=fontsize)
    tick_values=np.linspace(Minimum,Maximum,5)
    cbar_ax2.yaxis.set_ticks(tick_values,np.round(tick_values,2))

    #tick_positions = np.linspace(Minimum, Maximum, 4)  # 4 tick positions
    #cbar_ax2.yaxis.set_ticks(tick_positions)  # Set tick locations
    #cbar_ax2.yaxis.set_ticklabels([f"{t:.1f}" for t in tick_positions])  # Format labels (1 decimal)


    plt.tight_layout(rect=[0.1, 0, .9, 1])
    if filename=="":
        plt.show()
    else:
        plt.savefig(filename)

def plot_curvature(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Plot the curvature of a membrane",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d','--numpys_directory',type=str,help="Specify the path to the numpy direcory. This would coincde with the out folder from calculate.")
    parser.add_argument('-l1','--layer1',type=str,default="Upper",help="Custom name for layer 1.")
    parser.add_argument('-l2','--layer2',type=str,default="Lower",help="Custom name for layer 2.")
    parser.add_argument('-l3','--layer3',type=str,default="Both",help="Custom name for layer 3. Layer 3 is the base line for the Z fitting plot")
    parser.add_argument('--minimum',type=float,default=None,help="Supply a custom colorbar value for the curvature plots (minimum)")
    parser.add_argument('--maximum',type=float,default=None,help="Supply a custom colorbar value for the curvature plots (maximum)")
    parser.add_argument('-o','--outfile',type=str,default="",help="Specify the path to save the image, if none is given, image is shown.")
    parser.add_argument('-r','--rotation',type=bool,default=False,help="Specify if each frame's fourier transform should be rotated such that each frame's protein has the same orientation")
   
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    if args.minimum is not None and args.maximum is not None:
        minmax=[args.minimum,args.maximum]
    else:
        minmax=None

    try:
        draw(Dir=args.numpys_directory,layer1=args.layer1,layer2=args.layer2,layer3=args.layer3,minmax=minmax,filename=args.outfile,rotate=args.rotation)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

