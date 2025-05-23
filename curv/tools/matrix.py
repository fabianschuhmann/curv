import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
from ..core.fourier_core import Fourier_Series_Function
from ..core.matrix_core import Matrix_Function
import os
import glob 

logger = logging.getLogger(__name__)



def calc(in_dir,out_dir,Nx,Ny):

    """
    in_dir: string where the input data resides
    out_dir: string where to save the output
    Nx:int, number of modes in X direction
    Ny:int, number of modes in Y direction
    """

    fourier_data=[]
    for file_path in glob.glob(in_dir+f"Anm_*.npy"):
        fourier_data.append(np.load(file_path))
    box_size = np.load(in_dir+"boxsize.npy")
    fourier_data = np.asarray(fourier_data)
    matrix_function=Matrix_Function(box_size[0],box_size[1],Nx,Ny)
    A_vector = np.mean(fourier_data,axis=0)   
    A_matrix = np.mean(
        fourier_data[:, :, None] * fourier_data[:, None, :],
        axis=0
    )  # shape: (n_coeffs, n_coeffs)
    #save this
    matrix_function.get_A_vector(A_vector)
    matrix_function.get_A_matrix(A_matrix)
    matrix_function.make_sigmaA_matrix()
    matrix_function.make_q_vector()
    matrix_function.make_Hm_matrix()


def matrix(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Calculate the curvature of a membrane",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Specify a path to a folder where the input data resides")
    parser.add_argument('-o', '--out', type=str, required=True,
                        help="Specify a path to the output folder")
    parser.add_argument('-nx', '--nx', type=int, required=True,
                        help="Number of modes in X direction")
    parser.add_argument('-ny', '--ny', type=int, required=True,
                        help="Number of modes in Y direction")
    parser.add_argument('--clear', action='store_true',
                        help="Clear existing .npy files in the output folder")

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
        calc(args.input,args.out,args.nx,args.ny)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise