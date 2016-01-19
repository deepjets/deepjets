import numpy as np
import matplotlib.pyplot as plt
from deepjets.samples import get_sample

images, pt, weights = get_sample('w.config', 500,
                                 pt_min=200, pt_max=500, pt_bins=10)

# plot unweighted and weighted pT
fig = plt.figure(figsize=(5, 5))
ax  = fig.add_subplot(111)
ax.hist(pt, bins=np.linspace(200, 500, 51),
        histtype='stepfilled', facecolor='none', edgecolor='blue')
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
fig.tight_layout()
plt.savefig('pt.png')

fig = plt.figure(figsize=(5, 5))
ax  = fig.add_subplot(111)
ax.hist(pt, bins=np.linspace(200, 500, 51), weights=weights,
        histtype='stepfilled', facecolor='none', edgecolor='blue')
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
fig.tight_layout()
plt.savefig('pt_weighted.png')
