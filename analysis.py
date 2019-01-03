import os

os.system("python -m cProfile -o profile_data.pyprof run_and_compare.py inputs.txt")

os.system("pyprof2calltree -i profile_data.pyprof")

os.system("kcachegrind profile_data.pyprof.log")
