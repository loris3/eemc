import sys
from .run_gradsim import main

def wrapper():
    args = sys.argv[1:]  # take all command line arguments
    args.extend(["--mode", "normalized"])
    return main(args)

if __name__ == "__main__":
    wrapper()