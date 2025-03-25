import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
from ..core.fourier_core import Fourier_Series_Function
from scipy.spatial import distance_matrix as dm
import networkx as nx
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_rotation_angles(out_dir,u,From,Until,Step,sele):
    u.trajectory[0]
    Lx, Ly, Lz = u.dimensions[:3]
    side_of_box = np.array([Lx, Ly/2, Lz/2])
    selection=u.select_atoms(sele)
    com_selection = selection.center_of_mass()
    #dist_to_x_edge = min(com_selection[0], Lx - com_selection[0])
    #dist_to_y_edge = min(com_selection[1], Ly - com_selection[1])
    dist_to_x_edge = Lx/2
    dist_to_y_edge = Ly/2
    radius_threshold = min(dist_to_x_edge, dist_to_y_edge)
    
    o_list = []
    p_list = []
    for ts in tqdm(u.trajectory[From:Until:Step]):
        selection=u.select_atoms(sele)
        com_selection = selection.center_of_mass()
        o_list.append(com_selection)
        
        distances = np.linalg.norm(selection.positions[:,:2] - com_selection[:2])
        farthest_atom_index = np.argmax(distances)
        farthest_from_selection = selection.positions[farthest_atom_index]
        p_list.append(farthest_from_selection)

    #df = pd.DataFrame({'o': o_list, 'p': p_list})
    o_array = np.array(o_list)
    p_array = np.array(p_list)
    result_array = np.column_stack((o_array, p_array))
    
    np.save(file = f'{out_dir}rotation_vectors_o', arr = o_array)
    np.save(file = f'{out_dir}rotation_vectors_p', arr = p_array)

def calc_vectors(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Write an index file to be used for other curv tasks",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file")
    parser.add_argument('-F','--From',default=0,type=int,help="Discard all frames in the trajectory prior to the frame supplied here")
    parser.add_argument('-U','--Until',default=None,type=int,help="Discard all frames in the trajectory after to the frame supplied here")
    parser.add_argument('-S','--Step',default=1,type=int,help="Traverse the trajectory with a step length supplied here")
    parser.add_argument('-o','--out',default="",type=str,help="Specify a path to the to written rotation vector file")
    parser.add_argument('-n','--selection',type=str,help="Sepcifies reference point for the rotation")
    
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        universe=mda.Universe(args.structure,args.trajectory)
        get_rotation_angles(out_dir=args.out,u=universe,From=args.From,Until = args.Until,Step=args.Step,sele=args.selection)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

