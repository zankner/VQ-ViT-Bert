import os
import numpy as np


def log(vals, meter_name, log_dir, ext=".npy"):
    with open(os.path.join(log_dir, meter_name, ext)) as f:
        np.save(vals, f)