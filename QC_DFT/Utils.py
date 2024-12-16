import numpy as np
import scipy


import numpy as np

class Gates:
    def I():
        return np.array([[1,0],[0,1]], dtype=complex)
    
    def X():
        return np.array([[0,1],[1,0]], dtype=complex)
    
    def Y():
        return np.array([[0,-1j],[1j,0]], dtype=complex)
    
    def Z():
        return np.array([[1,0],[0,-1]], dtype=complex)
    
    def H():
        return np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
    
    def CX():
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    
    def RX(theta):
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],[-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    
    def RY(theta):
        return np.array([[np.cos(theta / 2), -1 * np.sin(theta / 2)],[np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    
    def RZ(theta):
        return np.array([[np.exp(-1j * theta / 2), 0],[0, np.exp(1j * theta / 2)]], dtype=complex)
    
    def T():
        return np.array([[1,0],[0,np.exp(1j * np.pi / 4)]], dtype=complex)
    
    def TDG():
        return np.array([[1,0],[0,np.exp(-1j * np.pi / 4)]], dtype=complex)
    



def modify_string(s, i, new_char):
    if i < 0 or i >= len(s):
        raise ValueError("Index out of range")
    
    modified_string = s[:i] + new_char + s[i+1:]
    return modified_string

PAULI_I = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)

PAULI_X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

PAULI_Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

PAULI_Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

PAULI_Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

PAULI_2S = [
    np.kron(PAULI_I, PAULI_I),
    np.kron(PAULI_I, PAULI_X),
    np.kron(PAULI_I, PAULI_Y),
    np.kron(PAULI_I, PAULI_Z),
    np.kron(PAULI_X, PAULI_I),
    np.kron(PAULI_X, PAULI_X),
    np.kron(PAULI_X, PAULI_Y),
    np.kron(PAULI_X, PAULI_X),
    np.kron(PAULI_Y, PAULI_I),
    np.kron(PAULI_Y, PAULI_X),
    np.kron(PAULI_Y, PAULI_Y),
    np.kron(PAULI_Y, PAULI_Z),
    np.kron(PAULI_Z, PAULI_I),
    np.kron(PAULI_Z, PAULI_X),
    np.kron(PAULI_Z, PAULI_Y),
    np.kron(PAULI_Z, PAULI_Z),
]


CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

CNOT_DAG = np.transpose(np.conjugate(CNOT))


def generate_hermitian(u_params):
    hermitian = np.zeros((4, 4), dtype=complex)
    for i in range(16):
        hermitian = np.add(hermitian, u_params[i] * PAULI_2S[i])
    return hermitian

def generate_unitary(u_params):
    hermitian = generate_hermitian(u_params)
    eigenvalues, eigenvectors = np.linalg.eig(hermitian)

    eigenvalues = np.exp(eigenvalues * 1j)

    diagonal_matrix = np.diag(eigenvalues)

    unitary = eigenvectors @ diagonal_matrix @ np.transpose(np.conjugate(eigenvectors))

    return unitary