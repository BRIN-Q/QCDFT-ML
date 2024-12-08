from venv import create
from os import getcwd, mkdir
from os.path import join, abspath
from subprocess import run

try:
    mkdir("Results")
except:
    pass

try:
    mkdir("Models")
except:
    pass

dir = join(getcwd(), "qc-dft-venv")
create(dir, with_pip=True)

# where requirements.txt is in same dir as this script
run(["bin/pip", "install", "-r", abspath("requirements.txt")], cwd=dir)