from .run_gradsim import main as m

def main(args):
    args = list(args)
    if "--mode" in args:
        index = args.index("--mode")
        args[index + 1] = "normalized"
    else:
        args += ["--mode", "normalized"]

    return m(args)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])