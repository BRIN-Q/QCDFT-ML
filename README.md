# QC-DFT


This repository provides source code used to generate the data presented in "Efficient Classical Computation of Single-Qubit Marginal Measurement Probabilities to Simulate Certain Classes of Quantum Algorithms" paper.


# How to use

## Setup

- Clone the repository with the command:
```
git clone ....
```

- Move into the repository directory
```
cd ...
```

- Build the virtual environment and directories using BUILD.py. Ensure that you have at least Python 3.11 with pip and venv installed.
```
python BUILD.py
```


## Generating Figure 1 & S1
To generate Fig. 1 & Fig. S1 in the paper, we need to do the following steps:

- Generating neural network training data

  To do this, we simply need to run all of the cells in Generate-Training-Data.ipynb

- Creating and training the neural networks

  To do this, we simply need to run all of the cells in Train.ipynb. The resulting training data will be stored in the Results folder and can be plotted using Generate-Figures.ipynb to generate Fig. S1.


- Test the SQP error, mean fidelity, and mean SQP

  To do this, we simply need to run all of the cells in SQP.ipynb. The resulting data will be stored in the Results folder and can be plotted using Generate-Figures.ipynb to generate Fig. 1.


## Generating Table 1 & Table 2
To generate Table 1 & Table 2 in the paper, we simply need to run all of the cells in Grover.ipynb.


## Generating Figure 2
To generate Fig. 2 in the paper, we need to run Shor.py. To do this, first, we need to edit the python file to specify the number of threads to use by modifying the line below to the desired number of threads
```
thread_count = 16
```
Then, we should activate the virtual environment and run the python file by using these following commands (on Linux)
```
source qc-dft-venv/bin/activate
python Shor.py
``` 
The resulting data will be stored in the Results folder and can be plotted using Generate-Figures.ipynb to generate Fig. 2.

## Generating Figure S2
To generate Fig. S2 in the paper, we simply need to run all of the cells in Time.ipynb. The resulting data will be stored in the Results folder and can be plotted using Generate-Figures.ipynb to generate Fig. S2.