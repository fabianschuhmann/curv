import argparse
from importlib.metadata import version
import logging

from curv.tools.calculate import calculate
from curv.core.plot import main as plot
from curv.tools.write_ndx import write_ndx
from curv.tools.height import height




logger = logging.getLogger(__name__)

def run_module(module_name, args):
    """
    run the specified python module with given arguments.
    """
    if module_name == 'calculate':
        calculate(args)
    elif module_name == 'plot':
        plot(args)
    elif module_name == 'write_ndx':
        write_ndx(args)
    elif module_name == 'height':
        height(args)
    else:
        print(f"Unknown module: {module_name}")


def main():
    """
    main entry point for the curv command-line interface.
    """

    # define the python based modules
    modules = ['calculate', 'plot', 'write_ndx','height']

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
