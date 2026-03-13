import sys
from .run_embd import main

def wrapper():
    embd_model = "nvidia/llama-embed-nemotron-8b"
    args = sys.argv[1:]  # take all command line arguments
    args.extend(["--embd_model", embd_model])
    return main(args)

if __name__ == "__main__":
    wrapper()