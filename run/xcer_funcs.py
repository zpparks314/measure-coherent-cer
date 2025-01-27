import trueq as tq
from typing import List, Tuple, Union, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns
import trueq as tq
import trueq.math as tqm
import trueq.simulation as tqs
from scipy.linalg import expm
from scipy.optimize import curve_fit
from trueq.compilation import Compiler, CycleReplacement, MarkCycles


PAULI_DECAYS = ["X", "Y", "Z"]
NUMTOPAULI = {1: "X", 2: "Y", 3: "Z"}
PAULITONUM = {e: t for t, e in NUMTOPAULI.items()}

# Function to generate the KNR circuit collection


def generate_cer_circuits(cycle: tq.Cycle,  rep_len_tuples: List[Tuple[int, int]], n_circuits: int = 120):

    cer_circuits = tq.CircuitCollection()
    for (rep, seq_len) in rep_len_tuples:
        # makes a collection to be modified
        circuits = tq.make_knr(cycle,
                               seq_len,           # seq lengths
                               n_circuits,        # Pauli randomizations
                               # number of body-correlations
                               subsystems=tq.Subsystems([2]),
                               twirl=tq.Twirl("P", (0, 1, 2, 3, 4))
                               )

        # Construct a TrueQ compiler consisting of two passes:
        # CycleReplacement : replaces cycles with a given cycle
        # MarkCycles : marks cycles in the circuit (we mark them with repetition number)
        compiler = Compiler([CycleReplacement(target=cycle, replacement=rep * [cycle]),
                             MarkCycles(marker=rep)])

        # new collection to hold compiled circuits
        compiled_circuits = tq.CircuitCollection()

        for circuit in circuits:

            compiled_circuit = compiler.compile(circuit)

            compiled_circuits += compiled_circuit

        # update the keys of the compiled circuits
        compiled_circuits.update_keys(rep=rep)

        # add compiled circuits to the knr collection
        cer_circuits += compiled_circuits

    # shuffle the knr_circuits
    cer_circuits.shuffle()

    return cer_circuits


# Function to generate the Hamiltonian we will use as a noise model for the ancilla qubit
# The Hamiltonian is a dictionary with the keys "X", "Y", "Z" and the values are the angles of rotation in degrees
def generate_hamiltonian(xangle: float = 0.0, yangle: float = 0.0, zangle: float = 0.0):
    hamiltonian = {
        "X": xangle,
        "Y": yangle,
        "Z": zangle
    }
    return hamiltonian

# Function to generate a simulator with the given basic error rates and ancilla Hamiltonian and index
# The basic error rates are a dictionary with the keys "t1", "t2", "t_single", "t_entangling" and the values are the error rates obtained from IBM calibration data
# The ancilla Hamiltonian is a dictionary with the keys "X", "Y", "Z" and the values are the angles of rotation in degrees
# We use the ancilla Hamiltonian to add cycle noise to the simulator affecting the


def generate_simulator(basic_error_rates: Dict, ancilla_hamiltonian: Dict, ancilla_index: int, cycle: tq.Cycle, depol: float = None):

    # instantiate a simulator
    device = tqs.Simulator()

    # add the basic error rates
    device.add_relaxation(basic_error_rates["t1"], basic_error_rates["t2"],
                          basic_error_rates["t_single"], basic_error_rates["t_entangling"])

    # add the cycle noise to the simulator on the ancilla qubit
    device.add_cycle_noise(
        {(ancilla_index): tq.Gate.from_generators(
            {pauli: cst for pauli, cst in ancilla_hamiltonian.items()})},
        match=tqs.CycleMatch(cycle),
        cycle_offset=-1,
    )
    if depol is not None:
        device.add_depolarizing(depol)

    return device


def gaussian_random_walk(
    initial_angle: float, num_steps: int, step_size: float, std_dev: float, constant: bool = False
):
    angles = [initial_angle]
    if constant is True:
        angles = [initial_angle]*num_steps
    else:
        for i in range(num_steps - 1):
            step = np.random.normal(loc=step_size, scale=std_dev)
            angles.append(angles[-1] + step)
    return angles


def sample_gaussian(initial_angle: float, num_steps: int, std_dev: float):
    angles = np.random.normal(initial_angle, std_dev, size=num_steps)
    return angles

def generate_t2_values(initial_rate: float, num_steps: int, std_dev: float):
    return np.random.normal(initial_rate, initial_rate*std_dev, num_steps)


def laplace_random_walk(
    initial_angle: float, num_steps: int, step_size: float, std_dev: float
):
    angles = [initial_angle]
    for i in range(num_steps - 1):
        step = np.random.laplace(loc=step_size, scale=std_dev)
        angles.append(angles[-1] + step)
    return angles


def rate_to_angle(rate):
    return np.sqrt(rate)*(180/np.pi)*2

# convert an angle to a rate

def angle_to_rate(angle):
    return (angle/2)**2 * (np.pi/180)**2

# plot a gaussian distribution with the given mean, standard deviation and number of samples


def plot_gaussian(mean, std_dev, num_samples):
    samples = np.random.normal(mean, std_dev, size=num_samples)
    sns.histplot(samples, stat="density", kde=True)
    plt.show()

# plot the random walk of angles with the given initial angle, number of steps, step size and standard deviation


def plot_angle_walk(angle_list: List, initial_angle: float, num_batches: int, std_dev: float):
    sns.set_style("darkgrid")
    sns.set_context("paper")
    plt.plot(angle_list)
    plt.plot([initial_angle]*num_batches,
             color='black', label='Initial Angle (Z)')
    plt.plot([np.mean(angle_list)]*num_batches, color='black',
             linestyle="--", label='Average Angle (Z)')
    plt.title(
        F"Mean = {initial_angle:.5f}, Std Dev = {std_dev}")
    plt.xlabel('Step')
    plt.ylabel('Angle')
    plt.show()


def plot_TRF_region(sub_mat, sub_est, w, v, decoh_entangling, decoh_total, tw_z_err):
    sns.set()  # textwidth is 6.13899in
    sns.set_theme("notebook", palette="deep", style='darkgrid')

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6.13899, 6.13899/1.8))

    t = np.linspace(0, np.pi)
    x = (np.sqrt(w[0])*v[0, 0]*np.cos(t) +
        np.sqrt(w[1])*v[0, 1]*np.sin(t)) + sub_est[0]
    y1 = (np.sqrt(w[0])*v[1, 0]*np.cos(t) +
        np.sqrt(w[1])*v[1, 1]*np.sin(t))+sub_est[1]
    y2 = (y1[-1]-y1[0])/(x[-1]-x[0])*(x-x[0])+y1[0]

    plt.fill_between(x, y1, y2, alpha=0.4, linewidth=0)
    # =================================
    t = np.linspace(0, -np.pi)
    x = (np.sqrt(w[0])*v[0, 0]*np.cos(t) +
        np.sqrt(w[1])*v[0, 1]*np.sin(t)) + sub_est[0]
    y1 = (np.sqrt(w[0])*v[1, 0]*np.cos(t) +
        np.sqrt(w[1])*v[1, 1]*np.sin(t)) + sub_est[1]
    y2 = (y1[-1]-y1[0])/(x[-1]-x[0])*(x-x[0])+y1[0]

    plt.fill_between(x, y1, y2, alpha=0.4,  color='b', linewidth=0,
                    label='1$\sigma$ confidence region')
    # ================================
    t = np.linspace(0, 2*np.pi)
    x = (np.sqrt(w[0])*v[0, 0]*np.cos(t) +
        np.sqrt(w[1])*v[0, 1]*np.sin(t)) + sub_est[0]
    y = (np.sqrt(w[0])*v[1, 0]*np.cos(t) +
        np.sqrt(w[1])*v[1, 1]*np.sin(t)) + sub_est[1]
    plt.plot(x, y, 'b')

    plt.axvline(x=0, color='w', linewidth=3)
    plt.axhline(y=0, color='w', linewidth=3)

    plt.ylim(-0.001, 0.002)
    plt.xlim(-0.0001, 0.002)
    plt.plot(sub_est[0], sub_est[1], 'o', color='b',
            markersize=5, label='TRF Estimate')
    plt.plot(decoh_entangling[2].real/2, tw_z_err.real/2,
            '*', color='gold', markersize=10, label='Exact errors')

    plt.plot([0, decoh_total[2].real/2], [decoh_total[2].real/2, 0],
            '--', color='gold', label='Exact total error tradeoff line')

    plt.xlabel(r'Linearly growing Z error, $\left({{\rm lin}_Z}/{2}\right)$')
    plt.ylabel(r'Constant Z error, $\left({\rm cst}_Z / 2\right)$')
    plt.tight_layout()

    plt.legend()
# extract a results dataframe from a CER CircuitCollection (populated with results)


def extract_results_dataframe(cer_circuits: tq.CircuitCollection, rep_len_tuples: List[Tuple[int, int]], qubit_index: int = 2):
    rep_df_list = []
    n_random_cycles_df_list = []
    fid_df_list = []
    fid_std_df_list = []
    decay_list = []
    decay_num_list = []

    for decay in PAULI_DECAYS:
        for (rep, n_random_cycles) in rep_len_tuples:
            for mb in cer_circuits.subset(rep=rep, n_random_cycles=n_random_cycles).keys().measurement_basis:
                if str(mb)[qubit_index] == decay:
                    for circuit in cer_circuits.subset(rep=rep, n_random_cycles=n_random_cycles, measurement_basis=mb):
                        fid = 0
                        for outcome, count in circuit.results.items():
                            if (str(circuit.key.compiled_pauli)[qubit_index] == "I") | (str(circuit.key.compiled_pauli)[qubit_index] == decay):
                                if outcome[qubit_index] == '0':
                                    fid += count
                            elif outcome[qubit_index] == '1':
                                fid += count

                        fid_df_list.append(fid/circuit.results.n_shots*2-1)
                        fid_std_df_list.append(
                            2/np.sqrt(circuit.results.n_shots))
                        rep_df_list.append(rep)
                        n_random_cycles_df_list.append(n_random_cycles)
                        decay_list.append(decay)
                        decay_num_list.append(PAULITONUM[decay])

    d = {'fidelity': fid_df_list,
         'fidelity_std': fid_std_df_list,
         'rep': rep_df_list,
         'n_random_cycles': n_random_cycles_df_list,
         'Pauli': decay_list,
         'Pauli_num': decay_num_list
         }

    df = pd.DataFrame(data=d)
    return df

def extract_avg_dataframe(results_dataframe: pd.DataFrame, rep_len_tuples: List[Tuple[int, int]], confidence: float = 0.682):
    rep_df_list = []
    n_random_cycles_df_list = []
    fid_df_list = []
    fid_std_df_list = []
    decay_list = []
    decay_num_list = []
    for pauli in PAULI_DECAYS:
        for (rep, seq_len) in rep_len_tuples:
            fid_array = np.array(results_dataframe[(results_dataframe['Pauli'] == pauli) &
                                    (results_dataframe['n_random_cycles'] == seq_len) &
                                    (results_dataframe['rep'] == rep)]['fidelity'])
            n_samp = len(fid_array)
            m, se = np.mean(fid_array), scp.stats.sem(fid_array)
            h = se * scp.stats.t.ppf((1 + confidence) / 2., n_samp-1)

            fid_df_list.append(m)
            # Notice that we confound SE and STD here,
            # since we interepret SE as the STD on the sample mean estimator:
            fid_std_df_list.append(se)
            rep_df_list.append(rep)
            n_random_cycles_df_list.append(seq_len)
            decay_list.append(pauli)
            decay_num_list.append(PAULITONUM[pauli])

    d = {'fidelity': fid_df_list,
        'fidelity_std': fid_std_df_list,
        'rep': rep_df_list,
        'n_random_cycles': n_random_cycles_df_list,
        'Pauli': decay_list,
        'Pauli_num': decay_num_list
        }
    df_avg = pd.DataFrame(data=d)
    
    return df_avg

def extract_error_dataframe(cov_full, est_full):
    coh_error = []
    coh_error_std = []

    decoh_error = []
    decoh_error_std = []

    noncoh_error = []
    noncoh_error_std = []

    cst_error = []
    cst_error_std = []

    diff_error = []
    diff_error_std = []

    value_type = []
    decay = []

    fquad = []
    fquad_std = []

    flin = []
    flin_std = []

    fcst = []
    fcst_std = []

    a_param = []
    a_param_std = []

    decay_f = []

    """ 
    A_x  0, coh_x  1,  dcoh_x  2,  tw_offset_x  3,
    A_y  4, coh_y  5,  dcoh_y  6,  tw_offset_y  7,
    A_z  8, coh_z  9,  dcoh_z  10, tw_offset_z  11
    """

    for pauli in ['X', 'Y', 'Z']:

        # ---------- A --------------
        if pauli == 'X':
            val = est_full[0]  # value
            std = np.sqrt(cov_full[0, 0])  # std
        elif pauli == 'Y':
            val = est_full[4]  # value
            std = np.sqrt(cov_full[4, 4])  # std
        elif pauli == 'Z':
            val = est_full[0]  # value
            std = np.sqrt(cov_full[8, 8])  # std

        a_param.append(val)
        a_param_std.append(std)
        decay_f.append(pauli)

        # ---------- Cst fid component fit--------------
        if pauli == 'X':
            val = est_full[7]+est_full[11]  # value
            std = np.sqrt(cov_full[7, 7] + cov_full[11, 11] +
                        cov_full[7, 11]+cov_full[11, 7])  # std
        elif pauli == 'Y':
            val = est_full[3]+est_full[11]  # value
            std = np.sqrt(cov_full[3, 3] + cov_full[11, 11] +
                        cov_full[3, 11]+cov_full[11, 3])  # std
        elif pauli == 'Z':
            val = est_full[7]+est_full[3]  # value
            std = np.sqrt(cov_full[7, 7] + cov_full[3, 3] +
                        cov_full[7, 3]+cov_full[3, 7])  # std

        fcst.append(val)
        fcst_std.append(std)
        decay_f.append(pauli)

    # ---------- Quad fid component fit--------------
        if pauli == 'X':
            val = est_full[5]+est_full[9]  # value
            std = np.sqrt(cov_full[5, 5] + cov_full[9, 9] +
                        cov_full[5, 9]+cov_full[9, 5])  # std
        elif pauli == 'Y':
            val = est_full[1]+est_full[9]  # value
            std = np.sqrt(cov_full[1, 1] + cov_full[9, 9] +
                        cov_full[1, 9]+cov_full[9, 1])  # std
        elif pauli == 'Z':
            val = est_full[5]+est_full[1]  # value
            std = np.sqrt(cov_full[5, 5] + cov_full[1, 1] +
                        cov_full[5, 1]+cov_full[1, 5])  # std

        fquad.append(val)
        fquad_std.append(std)
        decay_f.append(pauli)

    # ---------- Lin fid component fit--------------
        if pauli == 'X':
            val = est_full[6]+est_full[10]  # value
            std = np.sqrt(cov_full[6, 6] + cov_full[10, 10] +
                        cov_full[6, 10]+cov_full[10, 6])  # std
        elif pauli == 'Y':
            val = est_full[2]+est_full[10]  # value
            std = np.sqrt(cov_full[2, 2] + cov_full[10, 10] +
                        cov_full[2, 10]+cov_full[10, 2])  # std
        elif pauli == 'Z':
            val = est_full[6]+est_full[2]  # value
            std = np.sqrt(cov_full[6, 6] + cov_full[2, 2] +
                        cov_full[6, 2]+cov_full[2, 6])  # std

        flin.append(val)
        flin_std.append(std)
        decay_f.append(pauli)

        # ---------- Coherent error contribution (quad/2)--------------
        if pauli == 'X':
            val = est_full[1]/2  # value
            std = np.sqrt(cov_full[1, 1])/2  # std
        elif pauli == 'Y':
            val = est_full[5]/2  # value
            std = np.sqrt(cov_full[5, 5])/2  # std
        elif pauli == 'Z':
            val = est_full[9]/2  # value
            std = np.sqrt(cov_full[9, 9])/2  # std

        coh_error.append(val)
        coh_error_std.append(std)
        value_type.append('Estimate')
        decay.append(pauli)

        # ---------- Decoherent error contribution (lin/2)--------------
        if pauli == 'X':
            val = est_full[2]/2  # value
            std = np.sqrt(cov_full[2, 2])/2  # std
        elif pauli == 'Y':
            val = est_full[6]/2  # value
            std = np.sqrt(cov_full[6, 6])/2  # std
        elif pauli == 'Z':
            val = est_full[10]/2  # value
            std = np.sqrt(cov_full[10, 10])/2  # std

        decoh_error.append(val)
        decoh_error_std.append(std)

        # ---------- Constant error contribution (cst/2) --------------
        if pauli == 'X':
            val = est_full[3]/2  # value
            std = np.sqrt(cov_full[3, 3])/2  # std
        elif pauli == 'Y':
            val = est_full[7]/2  # value
            std = np.sqrt(cov_full[7, 7])/2  # std
        elif pauli == 'Z':
            val = est_full[11]/2  # value
            std = np.sqrt(cov_full[11, 11])/2  # std

        cst_error.append(val)
        cst_error_std.append(std)

        # ---------- Non-coherent error contribution (lin+cst)/2--------------
        if pauli == 'X':
            val = (est_full[2]+est_full[3])/2  # value
            std = np.sqrt(cov_full[2, 2]+cov_full[2, 3] +
                        cov_full[3, 2]+cov_full[3, 3])/2  # std
        elif pauli == 'Y':
            val = (est_full[6]+est_full[7]) / 2  # value
            std = np.sqrt(cov_full[6, 6]+cov_full[6, 7] +
                        cov_full[7, 6]+cov_full[7, 7])/2  # std
        elif pauli == 'Z':
            val = (est_full[10]+est_full[11])/2  # value
            std = np.sqrt(cov_full[10, 10]+cov_full[10, 11] +
                        cov_full[11, 10]+cov_full[11, 11])/2  # std

        noncoh_error.append(val)
        noncoh_error_std.append(std)

        # ---------- (lin-cst)/2--------------
        if pauli == 'X':
            val = (est_full[2]-est_full[3])/2  # value
            std = np.sqrt(cov_full[2, 2]-cov_full[2, 3] -
                        cov_full[3, 2]+cov_full[3, 3])/2  # std
        elif pauli == 'Y':
            val = (est_full[6]-est_full[7]) / 2  # value
            std = np.sqrt(cov_full[6, 6]-cov_full[6, 7] -
                        cov_full[7, 6]+cov_full[7, 7])/2  # std
        elif pauli == 'Z':
            val = (est_full[10]-est_full[11])/2  # value
            std = np.sqrt(cov_full[10, 10]-cov_full[10, 11] -
                        cov_full[11, 10]+cov_full[11, 11])/2  # std

        diff_error.append(val)
        diff_error_std.append(std)


    error_dict = {'coh_error': coh_error,
                'coh_error_std': coh_error_std,
                'decoh_error': decoh_error,
                'decoh_error_std': decoh_error_std,
                'noncoh_error': noncoh_error,
                'noncoh_error_std': noncoh_error_std,
                'cst/2': cst_error,
                'cst/2_std': cst_error_std,
                '(lin-cst)/2': diff_error,
                '(lin-cst)/2_std': diff_error_std,
                'val_type': value_type,
                'Pauli': decay}

    error_table_dict = {'quad/2': coh_error,
                        'quad/2_std': coh_error_std,
                        'lin/2': decoh_error,
                        'lin/2_std': decoh_error_std,
                        '(lin+cst)/2': noncoh_error,
                        '(lin+cst)/2_std': noncoh_error_std,
                        'cst/2': cst_error,
                        'cst/2_std': cst_error_std,
                        '(lin-cst)/2': diff_error,
                        '(lin-cst)/2_std': diff_error_std,
                        'val_type': value_type,
                        'Pauli': decay}

    fid_table_dict = {'A': a_param,
                    'A_std': a_param_std,
                    'quad': fquad,
                    'quad_std': fquad_std,
                    'lin': flin,
                    'lin_std': flin_std,
                    'cst': fcst,
                    'cst_std': fcst_std,
                    'val_type': value_type,
                    'Pauli': decay}

    error_df_full = pd.DataFrame(data=error_dict)
    return error_df_full, error_table_dict, fid_table_dict

def batch_circuits(cer_circuits: tq.CircuitCollection, batch_size: int):
    batches = []
    for i in range(0, len(cer_circuits), batch_size):
        batches.append(cer_circuits[i : i + batch_size])
    print(f"Number of batches: {len(batches)}")
    print(f"Length of first batch: {len(batches[0])}")
    print(f"length of final batch: {len(batches[-1])}")
    return batches

# define a error model function with 12 parameters
# params: x, A_x, coh_x, dcoh_x, tw_offset_x, A_y, coh_y, dcoh_y, tw_offset_y, A_z, coh_z, dcoh_z, tw_offset_z
# also defines three functions for use in the model: delta_x, delta_y, delta_z

def error_model(x, A_x, coh_x, dcoh_x, tw_offset_x, A_y, coh_y, dcoh_y, tw_offset_y, A_z, coh_z, dcoh_z, tw_offset_z):
    def delta_x(idx_array):
        return np.array([int(idx == 1) for idx in idx_array])
    def delta_y(idx_array):
        return np.array([int(idx == 2) for idx in idx_array])
    def delta_z(idx_array):
        return np.array([int(idx == 3) for idx in idx_array])
    
    m, n, pauli_num = x

    fx = A_x*(1-(coh_y+coh_z)*n**2-(dcoh_y+dcoh_z)
              * n-(tw_offset_y+tw_offset_z))**m
    fy = A_y*(1-(coh_x+coh_z)*n**2-(dcoh_x+dcoh_z)
              * n-(tw_offset_x+tw_offset_z))**m
    fz = A_z*(1-(coh_y+coh_x)*n**2-(dcoh_y+dcoh_x)
              * n-(tw_offset_y+tw_offset_x))**m
    return fx*delta_x(pauli_num) + fy*delta_y(pauli_num)+fz*delta_z(pauli_num)


def calculate_decoherence(basic_error_rates: Dict):
    sup_single = tqs.add_basic.RelaxationNoise.relaxation_channel(
        basic_error_rates["t1"], basic_error_rates["t2"], basic_error_rates["t_single"], 0)

    tw_z_err = (
        (1+sup_single.ptm[3, 3]-sup_single.ptm[2, 2]-sup_single.ptm[1, 1])/2).real
    tw_x_err = (
        (1+sup_single.ptm[1, 1]-sup_single.ptm[2, 2]-sup_single.ptm[3, 3])/2).real
    tw_y_err = (
        (1+sup_single.ptm[2, 2]-sup_single.ptm[1, 1]-sup_single.ptm[3, 3])/2).real

    sup = tqs.add_basic.RelaxationNoise.relaxation_channel(
        basic_error_rates["t1"], basic_error_rates["t2"], basic_error_rates["t_entangling"], 0)

    z_err = (1+sup.ptm[3, 3]-sup.ptm[2, 2]-sup.ptm[1, 1])/2
    x_err = (1+sup.ptm[1, 1]-sup.ptm[2, 2]-sup.ptm[3, 3])/2
    y_err = (1+sup.ptm[2, 2]-sup.ptm[1, 1]-sup.ptm[3, 3])/2

    decoh_entangling = [x_err, y_err, z_err]
    decoh_total = [x_err+tw_x_err, y_err+tw_y_err, z_err+tw_z_err]
    
    return decoh_entangling, decoh_total, tw_x_err, tw_y_err, tw_z_err