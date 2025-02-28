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

def rotation_matrix(theta):
    theta_rad = np.radians(theta)
    return np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                     [np.sin(theta_rad), np.cos(theta_rad)]])

def rotation_logic(pre_rotation, u):
    curvature_info = pre_rotation
    #u.trajectory[len(u.trajectory)-1]
    box_size=u.dimensions[:3]
    side_of_box = np.array([box_size[0], box_size[1]/2, box_size[2]/2])

    protein = u.select_atoms("protein")
    com_protein = protein.center_of_mass()

    Lx, Ly, Lz = u.dimensions[:3]
 
    dist_to_x_edge = min(com_protein[0], Lx - com_protein[0])
    dist_to_y_edge = min(com_protein[1], Ly - com_protein[1])
    radius_threshold = min(dist_to_x_edge, dist_to_y_edge)

    # Save to a text file
    with open('rotation.txt', 'w') as f:
        f.write(f"{radius_threshold}\n")

    # Generate the grid of x, y positions corresponding to the curvature matrix
    x_coords, y_coords = np.meshgrid(
        np.linspace(-side_of_box[0] / 2, side_of_box[0] / 2, curvature_info.shape[1]),
        np.linspace(-side_of_box[1] / 2, side_of_box[1] / 2, curvature_info.shape[0])
    )

    # Find farthest away atom from protein center
    distances = np.linalg.norm(protein.positions - com_protein, axis=1)
    farthest_atom_index = np.argmax(distances)

    # Prepare vectors
    origin = protein.center_of_mass()
    point2 = protein.positions[farthest_atom_index]
    ba = point2 - origin  # Vector from `o` to `p2`
    bc = side_of_box - origin  # Vector from `o` to reference side of the box

    # Project vectors onto XY-plane
    ba_2 = np.array([ba[0], ba[1]])
    bc_2 = np.array([bc[0], bc[1]])

    # Compute rotation angle
    cos_theta = np.dot(ba_2, bc_2) / (np.linalg.norm(ba_2) * np.linalg.norm(bc_2))
    cos_theta = np.clip(cos_theta, -1, 1)  # Avoid floating-point errors
    theta = np.degrees(np.arccos(cos_theta))

    # Compute rotation matrix
    R = rotation_matrix(-1*theta)

    # Copy curvature matrix
    rotated_curvature = curvature_info.copy()

    # Calculates the radius threshold for the rotation
    # Need this logic so that the rotations works for square and rectangles
    
    # Matrix Rotation Loop
    for i in range(curvature_info.shape[0]):
        for j in range(curvature_info.shape[1]):
            x, y = x_coords[i, j], y_coords[i, j]
            distance = np.sqrt(x**2 + y**2)  # Distance from the center
            if distance <= radius_threshold:
                rotated_x, rotated_y = np.dot(R, np.array([x, y]))  # Rotate (x, y)

                # Find closest indices in the original grid
                i_new = np.argmin(np.abs(y_coords[:, 0] - rotated_y))
                j_new = np.argmin(np.abs(x_coords[0, :] - rotated_x))

                # Find closest indices in the original grid
                i_new = np.argmin(np.abs(y_coords[:, 0] - rotated_y))
                j_new = np.argmin(np.abs(x_coords[0, :] - rotated_x))
                # Assign rotated values while keeping boundaries
                if 0 <= i_new < curvature_info.shape[0] and 0 <= j_new < curvature_info.shape[1]:
                    rotated_curvature[i_new, j_new] = curvature_info[i, j]

    return rotated_curvature

def calc(out_dir,u,ndx,From=0,Until=None,Step=1,layer_string="Both", rotate = False):
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

                else:
                    fourier=fourier_by_layer(layer_group,box_size)
                    Z_fitted = np.array([fourier.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
                curvature = fourier.Curv(X, Y)
                coordinates = np.vstack([X.flatten(), Y.flatten(), Z_fitted.flatten()]).T
                Z_fitted=Z_fitted / 10 #Scale from \AA to nm
                np.save(f"{out_dir}/Z_fitted_{count}_{Layer}.npy",Z_fitted)

                if rotate == True:
                    Z_fitted_rotation = rotation_logic(Z_fitted,u)
                    np.save(f"{out_dir}/Z_fitted_rotation_{count}_{Layer}.npy",Z_fitted_rotation)

                curvature = curvature * 10 #Scale from \AA to nm
                np.save(f"{out_dir}/curvature_frame_{count}_{Layer}.npy", curvature)

                if rotate == True:
                    curvature_rotation = rotation_logic(curvature,u)
                    np.save(f"{out_dir}/curvature_rotation_{count}_{Layer}.npy", curvature_rotation)

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
    parser.add_argument('-r','--rotation',type=bool,default=False,help="Specify if each frame's fourier transform should be rotated such that each frame's protein has the same orientation")
    
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

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
        calc(out_dir=args.out,u=universe,ndx=args.index,From=args.From,Until=args.Until,Step=args.Step,layer_string=args.leaflet,rotate=args.rotation)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
