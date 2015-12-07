import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_jet_image(pixels, etaphi_range=(-1.25, 1.25, -1.25, 1.25),
                   filename='jet_image.png'):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    p = ax.imshow(pixels, extent=etaphi_range,
                  interpolation='none',
                  norm=LogNorm(vmin=1e-9, vmax=1e3))
    fig.colorbar(p, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()
