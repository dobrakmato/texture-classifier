import numpy as np
import matplotlib.pyplot as plt
IMAGE_SIZE = 64

X = np.load(f'./imgs_{IMAGE_SIZE}_3.npy', allow_pickle=True)
Y = np.load(f'./labels_{IMAGE_SIZE}_3.npy', allow_pickle=True)

print(X.shape)
print(Y.shape)

columns = 5
rows = 5
fig = plt.figure(figsize=(12, 12))
for i in range(1, columns * rows + 1):
    idx = np.random.randint(0, len(X))
    img = X[idx]
    labels = Y[idx]
    fig.add_subplot(rows, columns, i)
    plt.title(', '.join(labels), fontsize=6)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
    plt.imshow(np.float32(img), interpolation='nearest')
fig.tight_layout()
plt.show()
