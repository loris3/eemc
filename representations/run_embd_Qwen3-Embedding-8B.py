from .run_embd import main as m

def main(args):
    args = list(args)

    if "--embd_model" in args:
        index = args.index("--embd_model")
        args[index + 1] = "Qwen/Qwen3-Embedding-8B"
    else:
        args += ["--embd_model", "Qwen/Qwen3-Embedding-8B"]

    return m(args)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])