import sys
from .run_embd import main

def wrapper():
    embd_model = "tencent/KaLM-Embedding-Gemma3-12B-2511"
    args = sys.argv[1:]  # take all command line arguments
    args.extend(["--embd_model", embd_model])
    return main(args)

if __name__ == "__main__":
    wrapper()