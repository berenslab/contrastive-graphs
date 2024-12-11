import importlib
import sys

# import nik_graphs


def main():
    print(sys.argv)

    mod = importlib.import_module(".mnist", package="nik_graphs")
    mod.run_path("", "outfile_unused")


if __name__ == "__main__":
    main()
