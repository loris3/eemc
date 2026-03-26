from .run_embd import main as m

def main(args):
    args = list(args)

    if "--embd_model" in args:
        index = args.index("--embd_model")
        args[index + 1] = "tencent/KaLM-Embedding-Gemma3-12B-2511"
    else:
        args += ["--embd_model", "tencent/KaLM-Embedding-Gemma3-12B-2511"]

    return m(args)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])