import numpy as np
import convolutionalsparsecoding_1D as CSC
import time
import mne
import os.path as op
from matplotlib import pyplot as plt

data_path = op.join(mne.datasets.sample.data_path(), 'MEG',
                    'sample', 'sample_audvis_raw.fif')
raw = mne.io.RawFIF(data_path, preload=True, verbose=False)

picks = mne.pick_types(raw.info, meg=False, eeg=True)

data, times = raw[picks, :15000]

start = time.clock()

b = data

b = b - np.mean(b, axis=1)[:, None]
b /= b.std()

size_kernel = [16, 11]
lambda_residual = 1.0
lambda_prior = 1.0

data = data[:, ::4]

# Optim options
max_it = 50
tol = 1e-3

d, z, Dz = CSC.learn_conv_sparse_coder(b, size_kernel, lambda_residual, lambda_prior, max_it, tol)
