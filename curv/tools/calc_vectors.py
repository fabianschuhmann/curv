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

def get_rotation_angles(out_dir,u,From,Until,Step,sele1, sele2):
    o_list = []
    p_list = []
    for ts in tqdm(u.trajectory[From:Until:Step]):
        selection1=u.select_atoms(sele1)
        selection2=u.select_atoms(sele2)
        com_selection1 = selection1.center_of_mass()[:2]
        com_selection2 = selection2.center_of_mass()[:2]
        o_list.append(com_selection1)
        p_list.append(com_selection2)

    o_array = np.array(o_list)
    p_array = np.array(p_list)
    
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
    parser.add_argument('-p1','--selection1',type=str,help="Sepcifies reference point 1 for the selection")
    parser.add_argument('-p2','--selection2',type=str,help="Sepcifies reference point 2 for the selection")
    
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        universe=mda.Universe(args.structure,args.trajectory)
        get_rotation_angles(out_dir=args.out,u=universe,From=args.From,Until = args.Until,Step=args.Step,sele1=args.selection1, sele2=args.selection2)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
