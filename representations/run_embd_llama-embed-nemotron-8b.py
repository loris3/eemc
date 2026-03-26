from .run_embd import main as m

def main(args):
    args = list(args)

    if "--embd_model" in args:
        index = args.index("--embd_model")
        args[index + 1] = "nvidia/llama-embed-nemotron-8b"
    else:
        args += ["--embd_model", "nvidia/llama-embed-nemotron-8b"]

    return m(args)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])