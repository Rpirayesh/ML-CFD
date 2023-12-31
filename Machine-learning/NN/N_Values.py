import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from scoring import *
import torch
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score


def nested_defaultdict():
    return defaultdict(nested_defaultdict)

config_dict = nested_defaultdict()



def _get_network_type(weight_dict):
    if any(x.startswith('s_layer_') for x in weight_dict):
        if any(x.startswith('output') for x in weight_dict):
            network = 'transfer'
        else:
            network = 'multitask'
    else:
        network = 'baseline'
    return network


def _get_layer_key(layer):
    ordinal = 0 if layer.startswith('s_layer') else 2 if 'output' in layer else 1
    try:
        num = int(layer.split('_')[-1])
    except ValueError:
        num = 0
    return ordinal, num


def _get_layers(weight_dict, network, robot_name):
    if network == 'multitask':
        keys = (k for k in weight_dict if (k.startswith('s_layer_') or k.startswith(robot_name)))
    else:
        keys = weight_dict
    return list(sorted((x.split('/')[0] for x in keys), key=_get_layer_key))


def _relu(x):
    x[x < 0] = 0
    return x


def _linear(x):
    return x


def calc_layer_outputs(weight_dict, bias_dict, inp, robot_name, network=None):
    if network is None:
        network = _get_network_type(weight_dict)
    layers = _get_layers(weight_dict, network, robot_name)
    print(layers)
    wkeys = [l + '/weights' for l in layers]
    bkeys = [l + '/bias' for l in layers]
    activation = [_linear if 'output' in l else _relu for l in layers]
    x = inp
    results = dict()
    for l, w, b, sigma in zip(layers, wkeys, bkeys, activation):
        w = weight_dict[w]
        b = bias_dict[b]
        x = sigma(np.matmul(x, w) + b)
        results[l] = x
    return results


def process_network(data):
    weight_dict = data[0]
    network = _get_network_type(weight_dict)
    bias_dict = data[1]
    robots = list(filter(lambda x: x != 'input', data[2]))

    results = dict()
    for robot in robots:
        if network == 'multitask':
            inp = data[2]['input'][robot]
        else:
            inp = data[2][robot]['input']
        results[robot] = calc_layer_outputs(weight_dict, bias_dict, inp, robot, network)
    return results


# The Metric
def procrustes(A, B):
    """
    Computes Procrustes distance bewteen representations A and B
    """
    A_sq_frob = np.sum(A ** 2)
    B_sq_frob = np.sum(B ** 2)
    nuc = np.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc

# The Metric
def procrustes_Torch(A, B):
    """
    Computes Procrustes distance between representations A and B
    """
    A_sq_frob = torch.sum(A ** 2)
    B_sq_frob = torch.sum(B ** 2)
    nuc = torch.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc
# Make sure you have a GPU available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU instead.")

def lin_cka_dist(A, B):
    """
    Computes Linear CKA distance bewteen representations A and B
    """
    similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
    normalization = np.linalg.norm(A @ A.T, ord="fro") * np.linalg.norm(
        B @ B.T, ord="fro"
    )
    return 1 - similarity / normalization

def Normal(A):
        # Subtract the mean value from each column
    mean_column = np.mean(A, axis=0)
    A_centered = A - mean_column

    # Compute the Frobenius norm
    frobenius_norm = np.linalg.norm(A_centered, 'fro')

    # Divide by the Frobenius norm to obtain the normalized representation A*
    A_normalized = A_centered / frobenius_norm
    return A_normalized

def Normal_Torch(A):
    # Subtract the mean value from each column
    mean_column = torch.mean(A, axis=0)
    A_centered = A - mean_column

    # Compute the Frobenius norm
    frobenius_norm = torch.linalg.norm(A_centered, 'fro')

    # Divide by the Frobenius norm to obtain the normalized representation A*
    A_normalized = A_centered / frobenius_norm
    return A_normalized
# implement it on the dictionary


# Compute mutual information
def compute_mutual_information(mat_A, mat_B):
    flat_A = mat_A.flatten()
    flat_B = mat_B.flatten()
    mi = mutual_info_score(flat_A, flat_B)
    return mi

# Compute normalized mutual information
def compute_normalized_mutual_information(mat_A, mat_B):
    flat_A = mat_A.flatten()
    flat_B = mat_B.flatten()
    nmi = normalized_mutual_info_score(flat_A, flat_B, average_method="arithmetic")
    return nmi

from scipy.stats import entropy


# num_bins=60
num_bins=200

def compute_joint_histogram(mat_A, mat_B, num_bins):
    hist, _, _ = np.histogram2d(mat_A.flatten(), mat_B.flatten(), bins=num_bins)
    return hist

def compute_normalized_mutual_information_large(mat_A, mat_B, num_bins=60):
    joint_hist = compute_joint_histogram(mat_A, mat_B, num_bins)
    joint_prob = joint_hist / np.sum(joint_hist)
    
    marginal_prob_A = np.sum(joint_prob, axis=1)
    marginal_prob_B = np.sum(joint_prob, axis=0)

    ent_A = entropy(marginal_prob_A, base=2)
    ent_B = entropy(marginal_prob_B, base=2)
    
    joint_ent = entropy(joint_prob.flatten(), base=2)
    
    mi = ent_A + ent_B - joint_ent
    nmi = mi / np.sqrt(ent_A * ent_B)
    return nmi





# with open('F_dict.pkl', 'rb') as f:
#      PA_dict = pickle.load(f)


import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import nquad



def joint_probability_density(x, y, kde_A, kde_B):
    return kde_A(x) * kde_B(y)

def mutual_information_K(mat_A, mat_B):
    kde_A = gaussian_kde(mat_A.flatten())
    kde_B = gaussian_kde(mat_B.flatten())
    
    # Define the integration limits
    x_min, x_max = np.min(mat_A), np.max(mat_A)
    y_min, y_max = np.min(mat_B), np.max(mat_B)
    
    # Compute the joint probability density
    joint_density = lambda x, y: joint_probability_density(x, y, kde_A, kde_B)
    
    # Compute the mutual information
    integrand = lambda x, y: joint_density(x, y) * np.log2(joint_density(x, y) / (kde_A(x) * kde_B(y)))
    mi, _ = nquad(integrand, [[x_min, x_max], [y_min, y_max]])
    
    return mi

def entropy_K(mat):
    kde = gaussian_kde(mat.flatten())
    x_min, x_max = np.min(mat), np.max(mat)
    
    integrand = lambda x: kde(x) * np.log2(kde(x))
    ent, _ = nquad(integrand, [[x_min, x_max]])
    
    return -ent

def normalized_mutual_information_K(mat_A, mat_B):
    mi = mutual_information_K(mat_A, mat_B)
    ent_A = entropy_K(mat_A)
    ent_B = entropy_K(mat_B)
    
    nmi = mi / np.sqrt(ent_A * ent_B)
    return nmi

###############

import numpy as np
from sklearn.feature_selection import mutual_info_regression

# def compute_mi_R(A, B):
#     # Flatten matrices to 1-D arrays
#     A_flat = A.flatten()
#     B_flat = B.flatten()

#     # Reshape 1-D array to 2-D array because mutual_info_regression expects 2-D inputs for X
#     A_flat = A_flat[:, np.newaxis]

#     # Compute MI score
#     mi_score = mutual_info_regression(A_flat, B_flat, random_state=0)

#     # Since mutual_info_regression returns an array of MI values for each feature,
#     # and we have only one feature, we return the first (and only) MI value.
#     return mi_score[0]

import numpy as np
from sklearn.feature_selection import mutual_info_regression

def compute_mi_R(A, B, sample_ratio=0.6):
    # Flattening the matrices
    A_flat = A.flatten()
    B_flat = B.flatten()

    # Sampling the data
    sample_size = int(sample_ratio * len(A_flat))
    idx = np.random.choice(len(A_flat), size=sample_size, replace=False)

    A_sample = A_flat[idx]
    B_sample = B_flat[idx]

    print("A_sample=",A_sample,"B_sample=",B_sample)
    
    # Reshaping the data to fit the input format of mutual_info_regression
    A_sample = A_sample[:, np.newaxis]

    # Computing the mutual information
    mi = mutual_info_regression(A_sample, B_sample)
    
    return mi[0]
