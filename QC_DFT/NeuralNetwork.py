import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
import scipy.linalg
import torch.optim as optim

import numpy as np
import pennylane as qml
import os
import copy

from QC_DFT.Utils import PAULI_2S, CNOT, CNOT_DAG, generate_hermitian, generate_unitary

class QCDFTPytorchModel(nn.Module):
    def __init__(self):
        super(QCDFTPytorchModel, self).__init__()
        self.flatten_ = nn.Flatten()

        self.linears_ = nn.ModuleList()
        self.linears_.append(nn.Linear(16, 5, dtype=torch.float64))
        self.linears_.append(nn.Linear(5, 64, dtype=torch.float64))
        self.linears_.append(nn.Sigmoid())
        self.linears_.append(nn.Linear(64, 64, dtype=torch.float64))
        self.linears_.append(nn.Sigmoid())
        self.linears_.append(nn.Linear(64, 128, dtype=torch.float64))
        self.linears_.append(nn.Sigmoid())
        self.linears_.append(nn.Linear(128, 256, dtype=torch.float64))
        self.linears_.append(nn.Sigmoid())
        self.linears_.append(nn.Linear(256, 512, dtype=torch.float64))
        self.linears_.append(nn.Sigmoid())
        self.linears_.append(nn.Linear(512, 1024, dtype=torch.float64))
        self.linears_.append(nn.Sigmoid())
        self.linears_.append(nn.Linear(1024, 16, dtype=torch.float64))

    def forward(self, rho_control, rho_target):
        rho_c_r = torch.from_numpy(np.real(rho_control))
        rho_c_i = torch.from_numpy(np.imag(rho_control))
        rho_t_r = torch.from_numpy(np.real(rho_target))
        rho_t_i = torch.from_numpy(np.imag(rho_target))
        x = torch.cat((rho_c_r, rho_c_i, rho_t_r, rho_t_i), dim=1)
        x = self.flatten_(x)
        for linear in self.linears_:
            x = linear(x)
        return x

def fidelities(u_params, x_rho_c, x_rho_t, y_rho_big, ptrace_index):
    batch_size = u_params.shape[0]

    intermediary_unitary = np.array([generate_unitary(u_params[i]) for i in range(batch_size)])

    rho_big_pred = np.array([CNOT @ intermediary_unitary[i] @ np.kron(x_rho_c[i], x_rho_t[i]) @ np.transpose(np.conjugate(intermediary_unitary[i])) @ CNOT_DAG for i in range(batch_size)])

    fidelity = np.zeros((batch_size,), dtype=np.float64)
    for i in range(batch_size):
        try:
            fidelity[i] = qml.math.fidelity(qml.math.partial_trace(y_rho_big[i], ptrace_index), qml.math.partial_trace(rho_big_pred[i], ptrace_index))
        except:
            print("passing2")
        pass

    return fidelity


def get_ptrace_index(control_or_target):
    ptrace_index = None
    if control_or_target == 'control':
        ptrace_index = [1]
    elif control_or_target == 'target':
        ptrace_index = [0]
    else:
        raise Exception(f"Not control or target! {control_or_target}")
    
    return ptrace_index

def generate_loss_function(control_or_target):
    ptrace_index = get_ptrace_index(control_or_target)
    
    class RMS1FLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u_params, rho_control, rho_target, y_rho_big):
            u_params = u_params.numpy()

            batch_size = u_params.shape[0]
            intermediary_unitary = np.array([generate_unitary(u_params[i]) for i in range(batch_size)])
            rho_big_pred = np.array([CNOT @ intermediary_unitary[i] @ np.kron(rho_control[i], rho_target[i]) @ np.transpose(np.conjugate(intermediary_unitary[i])) @ CNOT_DAG for i in range(batch_size)])

            fidelity = np.zeros((batch_size,), dtype=np.float64)
            for i in range(batch_size):
                try:
                    fidelity[i] = qml.math.fidelity(qml.math.partial_trace(y_rho_big[i], ptrace_index), qml.math.partial_trace(rho_big_pred[i], ptrace_index))
                except:
                    print("passing3")
                    pass

            ctx.save_for_backward(torch.from_numpy(u_params), torch.from_numpy(rho_control), torch.from_numpy(rho_target), torch.from_numpy(y_rho_big),
                                  torch.from_numpy(intermediary_unitary), torch.from_numpy(rho_big_pred), torch.from_numpy(fidelity))
            return torch.from_numpy(np.array(np.sqrt(np.mean(np.square(1.0 - fidelity)))))   # a single number (averaged loss over batch samples)

        @staticmethod
        def backward(ctx, grad_output):
            u_params, rho_c, rho_t, y_rho_big, intermediary_unitary, rho_big_pred, fidelity = ctx.saved_tensors
            u_params = u_params.numpy()
            rho_c = rho_c.numpy()
            rho_t = rho_t.numpy()
            y_rho_big = y_rho_big.numpy()
            intermediary_unitary = intermediary_unitary.numpy()
            rho_big_pred = rho_big_pred.numpy()
            fidelity = fidelity.numpy()

            batch_size = u_params.shape[0]

            drho_dtheta = np.zeros((batch_size, 16, 2, 2), dtype=complex)
            for i in range(batch_size):
                for j in range(16):
                    drho_dtheta[i][j] = 1j * (qml.math.partial_trace(CNOT @ intermediary_unitary[i] @ PAULI_2S[j] @ np.kron(rho_c[i], rho_t[i]) @ np.transpose(np.conjugate(intermediary_unitary[i])) @ CNOT_DAG, ptrace_index)
                                            - qml.math.partial_trace(CNOT @ intermediary_unitary[i] @ np.kron(rho_c[i], rho_t[i]) @ np.transpose(np.conjugate(intermediary_unitary[i])) @ PAULI_2S[j] @ CNOT_DAG, ptrace_index))


            rms1f = np.sqrt(np.mean(np.square(1. - fidelity)))

            dfidelity_dtheta = np.zeros((batch_size, 16), dtype=np.float64)
            for i in range(batch_size):
                sqrtm_ = scipy.linalg.sqrtm(qml.math.partial_trace(y_rho_big[i], ptrace_index))
                left_side_inv = scipy.linalg.inv(scipy.linalg.sqrtm(sqrtm_ @ qml.math.partial_trace(rho_big_pred[i], ptrace_index) @ sqrtm_))
                for j in range(16):
                    try:
                        dfidelity_dtheta[i][j] = 0.5 * np.real(np.trace(left_side_inv @ sqrtm_ @ drho_dtheta[i][j] @ sqrtm_))
                    except:
                        print("passing")
                        pass

            final_grad = np.zeros((batch_size, 16), dtype=np.float64)
            for i in range(batch_size):
                for j in range(16):
                    final_grad[i][j] = (fidelity[i] - 1.0) * dfidelity_dtheta[i][j]
                    
            return torch.from_numpy(final_grad), None, None, None
    
    return RMS1FLoss
    

def split_real_imag(vec):
    return np.real(vec), np.imag(vec)

class QCDFTModel():
    def create():
        model = QCDFTModel()
        model.control_nn = QCDFTPytorchModel()
        model.target_nn = QCDFTPytorchModel()
        return model

    def load_data(data_path):
        density_matrices_load = np.load(data_path)
        np.random.shuffle(density_matrices_load)

        x_rho_big = []
        x_rho_c = []
        x_rho_t = []
        y_rho_big = []
        y_rho_c = []
        y_rho_t = []
        for big_dm in density_matrices_load:
            x_rho_big += [big_dm]
            x_rho_c += [qml.math.partial_trace(big_dm, [1])]
            x_rho_t += [qml.math.partial_trace(big_dm, [0])]

            big_dm_y = CNOT @ big_dm @ CNOT_DAG

            y_rho_big += [big_dm_y]
            y_rho_c += [qml.math.partial_trace(big_dm_y, [1])]
            y_rho_t += [qml.math.partial_trace(big_dm_y, [0])]

        x_rho_big = np.array(x_rho_big)
        x_rho_c = np.array(x_rho_c)
        x_rho_t = np.array(x_rho_t)
        y_rho_big = np.array(y_rho_big)
        y_rho_c = np.array(y_rho_c)
        y_rho_t = np.array(y_rho_t)

        return x_rho_c, x_rho_t, y_rho_big

    def train(self, data, data_count, num_epochs, learning_rate=0.0001):
        data = list(data)
        data_temp = copy.deepcopy(data)
        for i in range(len(data_temp)):
            data_temp[i] = data_temp[i][:data_count]

        x_rho_c, x_rho_t, y_rho_big = data_temp

        data_temp = copy.deepcopy(data)
        for i in range(len(data_temp)):
            data_temp[i] = data_temp[i][data_count:2 * data_count]

        x_rho_c_val, x_rho_t_val, y_rho_big_val = data_temp

        history = {
            "target_losses" : [],
            "target_losses_val" : [],
            "control_losses" : [],
            "control_losses_val" : [],
            "target_fidelity" : [],
            "target_fidelity_val" : [],
            "control_fidelity" : [],
            "control_fidelity_val" : [],
        }

        control_or_target = "control"
        ptrace_index = get_ptrace_index(control_or_target) 
        optimizer = optim.SGD(self.control_nn.parameters(), lr=learning_rate, momentum=0.9)
        loss_fn_ = generate_loss_function(control_or_target)
        loss_fn = loss_fn_.apply

        for epoch in range(num_epochs):
            self.control_nn.eval()
            with torch.no_grad():
                u_params_val = self.control_nn(x_rho_c_val, x_rho_t_val)
                fidelity_val = np.mean(fidelities(u_params_val.detach().numpy(), x_rho_c_val, x_rho_t_val, y_rho_big_val, ptrace_index))
                loss_val = loss_fn(u_params_val, x_rho_c_val, x_rho_t_val, y_rho_big_val)
                history["control_losses_val"] += [loss_val.item()]
                history["control_fidelity_val"] += [fidelity_val]

            self.control_nn.train()
            optimizer.zero_grad()
            u_params = self.control_nn(x_rho_c, x_rho_t)
            fidelity = np.mean(fidelities(u_params.detach().numpy(), x_rho_c, x_rho_t, y_rho_big, ptrace_index))
            loss = loss_fn(u_params, x_rho_c, x_rho_t, y_rho_big)
            history["control_losses"] += [loss.item()]
            history["control_fidelity"] += [fidelity]
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Mean Fidelity: {fidelity}')


        control_or_target = "target"
        ptrace_index = get_ptrace_index(control_or_target) 
        optimizer = optim.SGD(self.target_nn.parameters(), lr=learning_rate, momentum=0.9)
        loss_fn_ = generate_loss_function(control_or_target) 
        loss_fn = loss_fn_.apply

        for epoch in range(num_epochs):
            self.target_nn.eval()
            with torch.no_grad():
                u_params_val = self.target_nn(x_rho_c_val, x_rho_t_val)
                fidelity_val = np.mean(fidelities(u_params_val.detach().numpy(), x_rho_c_val, x_rho_t_val, y_rho_big_val, ptrace_index))
                loss_val = loss_fn(u_params_val, x_rho_c_val, x_rho_t_val, y_rho_big_val)
                history["target_losses_val"] += [loss_val.item()]
                history["target_fidelity_val"] += [fidelity_val]

            self.target_nn.train()
            optimizer.zero_grad()
            u_params = self.target_nn(x_rho_c, x_rho_t)
            fidelity = np.mean(fidelities(u_params.detach().numpy(), x_rho_c, x_rho_t, y_rho_big, ptrace_index))
            loss = loss_fn(u_params, x_rho_c, x_rho_t, y_rho_big)
            history["target_losses"] += [loss.item()]
            history["target_fidelity"] += [fidelity]
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Mean Fidelity: {fidelity}')

        for key, val in history.items():
            history[key] = np.array(val)

        return history
        


    def save(self, directory_path):
        os.mkdir(directory_path)
        torch.save(self.control_nn, f"{directory_path}/control.pt")
        torch.save(self.target_nn, f"{directory_path}/target.pt")


    def load(directory_path):
        qcdftModel = QCDFTModel.create()
        qcdftModel.control_nn = torch.load(f"{directory_path}/control.pt", weights_only=False)
        qcdftModel.control_nn.eval()

        qcdftModel.target_nn = torch.load(f"{directory_path}/target.pt", weights_only=False)
        qcdftModel.target_nn.eval()

        return qcdftModel

    #single density matrix inputs
    def predict(self, x_rho_c, x_rho_t):
        res = []
        x_rho_c = np.array([x_rho_c])
        x_rho_t = np.array([x_rho_t])
        
        u_params_control = np.real(self.control_nn(x_rho_c, x_rho_t).detach().numpy())[0]
        unitary_control = generate_unitary(u_params_control)
        prediction_control = qml.math.partial_trace(CNOT @ unitary_control @ np.kron(x_rho_c, x_rho_t) @ np.transpose(np.conjugate(unitary_control)) @ CNOT_DAG, [1])


        u_params_target = np.real(self.target_nn(x_rho_c, x_rho_t).detach().numpy())[0]
        unitary_target = generate_unitary(u_params_target)
        prediction_target = qml.math.partial_trace(CNOT @ unitary_target @ np.kron(x_rho_c, x_rho_t) @ np.transpose(np.conjugate(unitary_target)) @ CNOT_DAG, [0])

        return np.array(prediction_control), np.array(prediction_target)
