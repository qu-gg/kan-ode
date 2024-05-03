"""
@file plotting.py

Holds general plotting functions for reconstructions of the bouncing ball dataset
"""
import numpy as np
import matplotlib.pyplot as plt


def show_sequences(seqs, preds, out_loc, num_out=None):
    plt.figure(0)
    if not isinstance(seqs, np.ndarray):
        seqs = seqs.cpu().numpy()
        preds = preds.cpu().numpy()

    if num_out is not None:
        seqs = seqs[:num_out]
        preds = preds[:num_out]

    figure, axis = plt.subplots(num_out, 1)

    for i in range(num_out):
        axis[i].plot(seqs[i])
        axis[i].plot(preds[i], '--')

    # Save to out location
    plt.savefig(out_loc)
    plt.close()
