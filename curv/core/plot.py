import argparse
from importlib.metadata import version
import logging
import sys
from curv.tools.plot_curvature import plot_curvature
from curv.tools.plot_height import plot_height





logger = logging.getLogger(__name__)

def run_module(module_name, args):
    """
    run the specified python module with given arguments.
    """
    if module_name == 'curvature':
        plot_curvature(args)
    elif module_name == 'height':
        plot_height(args)
    else:
        print(f"Unknown module: {module_name}")


def main(args=""):
    """
    main entry point for the curv command-line interface.
    """

    modules=['curvature','height']

    sys.argv=[""]+args
    # call the right subroutine based on the module type

        # parse arguments before a calling module
    parser = argparse.ArgumentParser(
        description='curv: curv can make an appropriate index, calculate the membrane curvature and provide a standard plot',
        prog='curv',
        formatter_class=argparse.RawTextHelpFormatter 
    )

    parser.add_argument(
        'module',
        choices=modules,
        help='choice of which module to run, run curv <module> -h for detailed help\n\
<write_ndx> prepares an index file\n\
<calculate> calculates the curvature\n\
<plot> plots the curvature data based on a directory with the files from curvature'
    )

    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='arguments for the chosen module'
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f'%(prog)s {version("curv")}'
    )

    args = parser.parse_args()

    # call the right subroutine based on the module type

    run_module(args.module, args.args)


if __name__ == '__main__':
    main()
