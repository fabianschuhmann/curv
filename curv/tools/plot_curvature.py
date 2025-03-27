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
from scipy.ndimage import rotate


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_XY(box_size):
    x = np.linspace(0, box_size[0], 100)
    y = np.linspace(0, box_size[1], 100)
    X, Y = np.meshgrid(x, y)
    return X,Y

def shift_x(matrix):
    return np.hstack((matrix[:, 1:], matrix[:, :1]))

def shift_y(matrix):
    return np.vstack((matrix[1:], matrix[:1]))

def recenter_matrix(matrix, O, P):
    # Calculate matrix dimensions
    matrix_width, matrix_height = matrix.shape

    # Calculate shift amounts to center O (center of mass)
    shift_x = int(round(matrix_width // 2 - O[0]))
    shift_y = int(round(matrix_height // 2 - O[1]))

    # Apply the same shift to both O and P to ensure they stay aligned
    new_O = (int(round(O[0] + shift_x)), int(round(O[1] + shift_y)))
    new_P = (int(round(P[0] + shift_x)), int(round(P[1] + shift_y)))  # Same shift applied to P

    # Print shift amounts and new positions for debugging
    print(f"Shift amounts: {shift_x}, {shift_y}")
    print(f"New O: {new_O}, New P: {new_P}")

    # Recenter the matrix by applying the calculated shift
    recentered_matrix = np.roll(matrix, shift_x, axis=1)
    recentered_matrix = np.roll(recentered_matrix, shift_y, axis=0)

    return recentered_matrix, new_O, new_P



def rotation_matrix(theta):
    theta_rad = np.radians(theta)
    return np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                     [np.sin(theta_rad), np.cos(theta_rad)]])

def rotation_logic(Dir, PATH, o, p, rt, bs):
    radius_threshold = rt
    curvature_data = []
    curvature_info = np.load(PATH)  # Load curvature data

    new_O, new_P = o, p  # Initialize new_O and new_P to default values

    curvature_info, new_O, new_P = recenter_matrix(curvature_info, o, p)
    new_O = new_O * (bs[:2]/100)
    new_P = new_P * (bs[:2]/100)
    
    box_size = bs
    side_of_box = np.array([box_size[0]/2, box_size[1]])

    new_O_arr = np.array(new_O)
    new_P_arr = np.array(new_P)

    ba = new_P_arr[:2] - new_O_arr[:2]  # Vector from `o` to `p2`
    bc = side_of_box - new_O_arr[:2]  # Vector from `o` to reference side of the box

    # Project vectors onto XY-plane
    ba_2 = np.array([ba[0], ba[1]])
    bc_2 = np.array([bc[0], bc[1]])
        
    # Compute rotation angle `theta`
    cos_theta = np.dot(ba_2, bc_2) / (np.linalg.norm(ba_2) * np.linalg.norm(bc_2))
    cos_theta = np.clip(cos_theta, -1, 1)  # Avoid floating-point errors
    theta = np.degrees(np.arccos(cos_theta))

    cross_product = ba_2[0] * bc_2[1] - ba_2[1] * bc_2[0]

    # Determine if rotation is clockwise or counterclockwise
    if cross_product > 0:
        theta = +theta  # Positive rotation
    elif cross_product < 0:
        theta = -theta  # Negative rotation

    return rotate(curvature_info, theta, reshape = False)
    
def proper_rotate(curvature_info, angle, fill_value=np.nan):
    """Rotate the entire matrix using scipy's rotate with boundary handling."""
    # Use scipy to rotate the matrix, preserving the size
    rotated_matrix = rotate(curvature_info, angle, reshape=True, cval=fill_value)
    
    return rotated_matrix

def draw(Dir,layer1="Upper",layer2="Lower",layer3="Both",minmax=None, rotation=False, filename=""):
    # Plots
    fontsize=24
    box_size=np.load(Dir+"boxsize.npy")
    X,Y = get_XY(box_size)
    
    if rotation == True:
        o_array = np.load(Dir+"rotation_vectors_o.npy")
        p_array = np.load(Dir+"rotation_vectors_p.npy")
        
    radius_threshold = min(np.max(X/2), np.max(Y/2))
    # Upper layer 
    curvature_data1=[]
    for file_path in glob.glob(Dir+f"curvature_frame_*_{layer2}.npy"):
        if rotation == True:
            frame_idx = int(file_path.split("_")[-2]) - 1
            curvature_data1.append(rotation_logic(Dir, file_path, o_array[frame_idx], p_array[frame_idx], radius_threshold, box_size))
        else:
            curvature_data1.append(np.load(file_path))
    curvature_data1 = np.asarray(curvature_data1)
    curvature_data1=np.mean(curvature_data1,axis=0)

    # Lower layer 
    curvature_data2=[]
    for file_path in glob.glob(Dir+f"curvature_frame_*_{layer1}.npy"):
        if rotation  == True:
            frame_idx = int(file_path.split("_")[-2]) - 1
            curvature_data2.append(rotation_logic(Dir, file_path, o_array[frame_idx], p_array[frame_idx], radius_threshold, box_size))
        else:
            curvature_data2.append(np.load(file_path))
    curvature_data2 = np.asarray(curvature_data2)
    curvature_data2=np.mean(curvature_data2,axis=0)

    # Middle layer
    curvature_data3=[]
    for file_path in glob.glob(Dir+f"curvature_frame_*_{layer3}.npy"):
        if rotation  == True:
            frame_idx = int(file_path.split("_")[-2]) - 1
            curvature_data3.append(rotation_logic(Dir, file_path, o_array[frame_idx], p_array[frame_idx], radius_threshold, box_size))
        else:
            curvature_data3.append(np.load(file_path))
    curvature_data3 = np.asarray(curvature_data3)
    curvature_data3=np.mean(curvature_data3,axis=0)

    catted=[curvature_data1,curvature_data2,curvature_data3]

    if minmax is None:
        Minimum, Maximum =np.min([np.min(x) for x in catted]),np.max([np.max(x) for x in catted])
    else:
        Minimum, Maximum = minmax[0],minmax[1]
        print(f"A custom minimum and maximum has been set to {Minimum} and {Maximum}.")
        bmin, bmax =np.min([np.min(x) for x in catted]),np.max([np.max(x) for x in catted])
        print(f"Automatically, the values would have been assigned to {np.round(bmin,3)} and {np.round(bmax,3)}")

    # Thickness
    Z_fitted=[]
    for file_path in glob.glob(Dir+f"Z_fitted_*_{layer3}.npy"):
        if rotation  == True:
            frame_idx = int(file_path.split("_")[-2]) - 1
            Z_fitted.append(rotation_logic(Dir, file_path, o_array[frame_idx], p_array[frame_idx], radius_threshold, box_size))
        else:
            Z_fitted.append(np.load(file_path))
    Z_fitted = np.asarray(Z_fitted)
    Z_fitted=np.mean(Z_fitted,axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(24,28))  # Create 1 row, 2 columns of subplots
    axes=axes.flatten()
    for ax in axes:
        ax.set_aspect(box_size[0]/box_size[1])
        ax.set_xticks([0,box_size[0]])
        ax.set_yticks([0,box_size[1]])
        ax.set_xticklabels(['0', 'L$_x$'], fontsize=fontsize)  # Increase font size
        ax.set_yticklabels(['0', 'L$_y$'], fontsize=fontsize)

    if rotation == True:
        matrix_size = 100
        matrix_data = np.random.rand(matrix_size, matrix_size)
        radius_threshold = 51
        x_coords, y_coords = np.meshgrid(np.linspace(-matrix_size / 2, matrix_size / 2, matrix_size), np.linspace(-matrix_size / 2, matrix_size / 2, matrix_size))
        distance_from_center = np.sqrt(x_coords**2 + y_coords**2)
        mask = distance_from_center <= radius_threshold
        masked_data = np.ma.masked_where(~mask, matrix_data)
        Z_fitted = np.ma.masked_where(~mask, Z_fitted)
        curvature_data1 = np.ma.masked_where(~mask, curvature_data1)
        curvature_data2 = np.ma.masked_where(~mask, curvature_data2)
        curvature_data3 = np.ma.masked_where(~mask, curvature_data3)

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
    parser.add_argument('-r','--rotation',type=bool,default=False,help="specify the suffix for the rotation vector files.")
    parser.add_argument('-o','--outfile',type=str,default="",help="Specify the path to save the image, if none is given, image is shown.")
   
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    if args.minimum is not None and args.maximum is not None:
        minmax=[args.minimum,args.maximum]
    else:
        minmax=None

    try:
        draw(Dir=args.numpys_directory,layer1=args.layer1,layer2=args.layer2,layer3=args.layer3,minmax=minmax,rotation=args.rotation,filename=args.outfile)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
