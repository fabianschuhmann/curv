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
from scipy.stats import norm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def draw(Dir,name,hunit,time):
    # Plots
    fontsize=24
    data=[]
    names=[]
    for file_path in glob.glob(Dir+"/"+name):
        data.append(np.load(file_path))
        names.append(file_path)
    data=np.asarray(data)
    print(data.shape)
    if time is None:
        time=data.shape[1]-1
 
    fig, ax = plt.subplots(1, 1, figsize=(24,28))  # Create 1 row, 2 columns of subplots
    #axes=axes.flatten()

    for i,line in enumerate(data):
        plot_data=line
        #plt.hist(z_values2[:, i],label=f"Residue {z_value_2_names[i]}",color=color_blind_colors[i],bins=bins,alpha=alpha,density=True)
        mu, std = norm.fit(plot_data)

        # Generate normal distribution curve
        x_vals = np.linspace(min(plot_data), max(plot_data), 1000)
        pdf_vals = norm.pdf(x_vals, mu, std)
        plt.plot(x_vals, pdf_vals, label=names[i].split("/")[-1]+f'\n(μ={mu:.2f}, σ={std:.2f})', linewidth=5)


    # Set axis labels
    ax.set_xlabel(f"Height ({hunit})", fontsize=14*2) # Increase x-axis label font size
    ax.set_ylabel("Density", fontsize=14*2)  # Increase y-axis label font size

    # Set title
    ax.set_title("Height relative to middle layer", fontsize=16*2)

    # Set tick label sizes
    ax.tick_params(axis='x', labelsize=12*2)
    ax.tick_params(axis='y', labelsize=12*2)

    # Set y-axis limit
    #ax.set_ylim(0)
    #ax.set_xlim(0,time)

    # Remove top and right spines (axis lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Increase axis line widths
    ax.spines['bottom'].set_linewidth(1.5*2)
    ax.spines['left'].set_linewidth(1.5*2)

    # Adjust title again (if needed)
    #ax.set_title(f'Z-Coordinates Over Time for Selections', fontsize=16*2)

    # Add legend
    ax.legend(loc='best', fontsize=12*2, ncol=2)
    plt.tight_layout(rect=[0.1, 0, .9, 1])
    plt.show()

def plot_height(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Plot the relative height of a previously calculated directory",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d','--numpys_directory',type=str,help="Specify the path to the numpy direcory. This would coincde with the out folder from height.")
    parser.add_argument('-n','--name',type=str,default="*.npy",help="Pattern of files to be considered in height folder, i.e height_*_Lower.npy")
    parser.add_argument('-t','--time',type=float,default=None,help="Set the time of the last frame of the calculated simulation")
    parser.add_argument('-hu','--height_unit',type=str,default="$\AA$",help="Set the height unit (default $\AA$)")

   
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        draw(Dir=args.numpys_directory,name=args.name,hunit=args.height_unit,time=args.time)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise