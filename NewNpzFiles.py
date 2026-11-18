import numpy as np

data = np.load("video-17-segment-1-frame-1.npz", allow_pickle=False)

print(data.files)  # names of the arrays inside

for name in data.files:
    arr = data[name]
    print(name, arr.shape, arr.dtype)