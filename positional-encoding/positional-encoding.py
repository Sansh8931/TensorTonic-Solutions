import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):

    positions = np.arange(seq_len)[:, np.newaxis]

    i = np.arange((d_model + 1)//2)[np.newaxis, :]

    div_term = np.power(base, (2 * i) / d_model)

    angles = positions / div_term

    pe = np.zeros((seq_len, d_model), dtype=float)

    pe[:, 0::2] = np.sin(angles)

    pe[:, 1::2] = np.cos(angles[:, :d_model//2])

    return pe