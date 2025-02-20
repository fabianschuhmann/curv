import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
from ..core.fourier_core import Fourier_Series_Function
from scipy.spatial import distance_matrix as dm
import networkx as nx

logger = logging.getLogger(__name__)


def get_components(matrix, threshold):
    init_threshold=np.percentile(matrix,threshold)
    adj_matrix=np.where(matrix>init_threshold,0,matrix)
    G=nx.from_numpy_array(adj_matrix)
    components=list(nx.connected_components(G))
    if len(components) > 2:
        return get_components(matrix,threshold*1.5)
    elif len(components) < 2:
        return get_components(matrix,threshold/2)
    else:
        return components

def write(out_dir,u,selection,flip=False):
    ndx={}
    if flip:
        firstname="Upper"
        secondname="Lower"
    else:
        firstname="Lower"
        secondname="Upper"
    selection=u.select_atoms(selection)
    d_matrix=dm(selection.atoms.positions,selection.atoms.positions)
    init_threshold=0.01
    two_components=get_components(d_matrix,init_threshold)
    first = [selection.atoms[x].index for x in two_components[0]]
    np.save(arr=np.asarray([selection.atoms[x].position for x in two_components[0]]),file="component1.npy")
    np.save(arr=np.asarray([selection.atoms[x].position for x in two_components[1]]),file="component2.npy")
    first_z = np.mean(np.asarray([selection.atoms[x].position[2] for x in two_components[0]]))
    second = [selection.atoms[x].index for x in two_components[1]]
    second_z=np.mean(np.asarray([selection.atoms[x].position[2] for x in two_components[1]]))
    with open(out_dir,"w",encoding="UTF8") as f:
        f.write(f"[ {firstname} ]; Avg. Z: {first_z}")
        ndx[firstname]=[]
        for i,index in enumerate(first):
            if i%16==0:
                f.write("\n")
            f.write(f"{index+1} ")
            ndx[firstname].append(index+1)
        f.write("\n")
        f.write(f"[ {secondname} ]; Avg. Z: {second_z}")
        ndx[secondname]=[]
        for i,index in enumerate(second):
            if i%16==0:
                f.write("\n")
            f.write(f"{index+1} ")
            ndx[secondname].append(index+1)
        f.write("\n")

    return ndx

def write_ndx(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Write an index file to be used for other curv tasks",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file")
    parser.add_argument('-n','--selection',type=str,help="Specify the selection of particles to be considered")
    parser.add_argument('-o','--out',default="monolayers.ndx",type=str,help="Specify a path to the to be written index file")
    parser.add_argument('-F','--flip',default=False,action='store_true',help="flip Upper and Lower index")
    
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        universe=mda.Universe(args.structure,args.trajectory)
        write(out_dir=args.out,u=universe,selection=args.selection,flip=args.flip)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise