import numpy as np

from QC_DFT.Utils import Gates
from pennylane.math import partial_trace, purity

class QCDFT_Circuit:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.operators = []

        self._reinitialize()

    def _reinitialize(self):
        self.rdms = np.array([np.array([[1, 0], [0, 0]], dtype=complex) for _ in range(self.n_qubits)], dtype=complex)
        self.operator_iterator = 0

    def add_operator(self, operator, qubit):
        self.operators += [(operator, qubit)]

    def i(self, qubit):
        self.add_operator(Gates.I(), qubit)

    def x(self, qubit):
        self.add_operator(Gates.X(), qubit)

    def y(self, qubit):
        self.add_operator(Gates.Y(), qubit)

    def z(self, qubit):
        self.add_operator(Gates.Z(), qubit)

    def h(self, qubit):
        self.add_operator(Gates.H(), qubit)

    def cx(self, control_qubit, target_qubit):
        self.add_operator(Gates.CX(), [control_qubit, target_qubit])

    def rx(self, theta, qubit):
        self.add_operator(Gates.RX(theta), qubit)
    
    def ry(self, theta, qubit):
        self.add_operator(Gates.RY(theta), qubit)

    def rz(self, theta, qubit):
        self.add_operator(Gates.RZ(theta), qubit)

    def t(self, qubit):
        self.add_operator(Gates.T(), qubit)

    def tdg(self, qubit):
        self.add_operator(Gates.TDG(), qubit)

    def toffoli(self, control_qubit1, control_qubit2, target_qubit):
        self.h(target_qubit)
        self.cx(control_qubit1, target_qubit)
        self.tdg(target_qubit)
        self.cx(control_qubit2, target_qubit)
        self.t(target_qubit)
        self.cx(control_qubit1, target_qubit)
        self.tdg(target_qubit)
        self.cx(control_qubit2, target_qubit)
        self.t(target_qubit)
        self.t(control_qubit1)
        self.h(target_qubit)
        self.cx(control_qubit2, control_qubit1)
        self.tdg(control_qubit1)
        self.t(control_qubit2)
        self.cx(control_qubit2, control_qubit1)

    def mcx(self, control_qubits, target_qubit):
        auxilliary_count = len(control_qubits) - 2

        control_auxilliary_qubits = control_qubits + [self.n_qubits + auxilliary_iterator for auxilliary_iterator in range(auxilliary_count)]

        auxilliary_iterator = self.n_qubits
        for i in range(0, len(control_auxilliary_qubits), 2):
            if i < len(control_auxilliary_qubits) - 2:
                self.toffoli(control_auxilliary_qubits[i], control_auxilliary_qubits[i + 1], auxilliary_iterator)
                auxilliary_iterator += 1
            else:
                self.toffoli(control_auxilliary_qubits[i], control_auxilliary_qubits[i + 1], target_qubit)

        auxilliary_iterator -= 1

        for i in range(len(control_auxilliary_qubits) - 4, -1, -2):
            self.toffoli(control_auxilliary_qubits[i], control_auxilliary_qubits[i + 1], auxilliary_iterator)
            auxilliary_iterator -= 1


    def evolve(self, step_count = None):
        if step_count == None:
            step_count = len(self.operators) - self.operator_iterator

        for _ in range(step_count):
            gate, qubit = self.operators[self.operator_iterator]
            self.operator_iterator += 1

            if isinstance(qubit, int):
                self.rdms[qubit] = gate @ self.rdms[qubit] @ np.conjugate(np.transpose(gate))
            else:
                big_rdm = np.kron(self.rdms[qubit[0]], self.rdms[qubit[1]])
                
                big_rdm = gate @ big_rdm @ np.conjugate(np.transpose(gate))

                self.rdms[qubit[0]] = partial_trace(big_rdm, [1])
                self.rdms[qubit[1]] = partial_trace(big_rdm, [0])
                
        return self.rdms[:self.n_qubits]


class NN_QCDFT_Circuit(QCDFT_Circuit): 
    
    def __init__(self, n_qubits, model): 
        super().__init__(n_qubits)
        self.model = model

    def evolve(self, step_count = None):
        if step_count == None:
            step_count = len(self.operators) - self.operator_iterator

        for _ in range(step_count):
            gate, qubit = self.operators[self.operator_iterator]
            self.operator_iterator += 1

            if isinstance(qubit, int):
                self.rdms[qubit] = gate @ self.rdms[qubit] @ np.conjugate(np.transpose(gate))
            else:
                control_rdm = self.rdms[qubit[0]]
                target_rdm = self.rdms[qubit[1]]
                if (purity(control_rdm, [0]) > 0.999 and purity(target_rdm, [0]) > 0.999):

                    big_rdm = np.kron(self.rdms[qubit[0]], self.rdms[qubit[1]])
                    
                    big_rdm = gate @ big_rdm @ np.conjugate(np.transpose(gate))

                    self.rdms[qubit[0]] = partial_trace(big_rdm, [1])
                    self.rdms[qubit[1]] = partial_trace(big_rdm, [0])

                else:
                    control_rdm, target_rdm = self.model.predict(self.rdms[qubit[0]], self.rdms[qubit[1]])                
                    self.rdms[qubit[0]] = control_rdm
                    self.rdms[qubit[1]] = target_rdm

        return self.rdms[:self.n_qubits]