# load the library - Pyswip 0.2.10 should use SWI Prolog version 8.4.2,
# or else you may face some errors
from pyswip import Prolog
from utils import join_path
import sys
import os


def compile_abc(home_dir: str, theory_dir: str, export_folder_dir: str) -> int:
    prolog = Prolog()
    # consult files
    prolog.consult(join_path([home_dir, "code", "main.pl"]))
    prolog.consult(join_path([home_dir, theory_dir]))

    # create the fold if necessary
    export_folder_abs_dir = join_path([home_dir, export_folder_dir])
    if not os.path.exists(export_folder_abs_dir):
        os.mkdir(export_folder_abs_dir)
    # modify the export directory
    [_ for _ in prolog.query(f"working_directory(_, '{export_folder_abs_dir}')")]

    # run the code
    fault_num = [soln for soln in prolog.query("abc(X)")][0]["X"]
    return fault_num


if __name__ == "__main__":
    home_dir, theory_dir, export_folder_dir = sys.argv[1:4]
    compile_abc(home_dir, theory_dir, export_folder_dir)