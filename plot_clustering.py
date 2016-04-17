
import matplotlib.pyplot as plt

fig, ax = plt.subplots(6, 5, figsize=(20, 22))

# plot R=1, sjR=0.5,4,3,2 with and without zooming

params_dict = {
    'PhaseSpace:pTHatMin': 200,
    'PhaseSpace:pTHatMax': 500}

# generate 10k events
from deepjets.generate import generate_events
events = list(generate_events('w.config', 10000, params_dict=params_dict))

from deepjets.clustering import cluster
from deepjets.preprocessing import pixel_edges, preprocess
from deepjets.utils import plot_jet_image

import numpy as np

edges = pixel_edges(jet_size=1.0, border_size=0)
zoom_edges = pixel_edges(jet_size=1.0, border_size=1.0)
Rs = (.5, .4, .3, .2, .1)
for i, R in enumerate(Rs):
    jets = list(cluster(events, jet_size=1.0, subjet_size=R))
    images = np.array([
        preprocess(jet.subjets, jet.trimmed_constit, edges)
        for jet in jets]).mean(axis=0)

    dr_pt = np.empty((len(jets), 2), dtype=np.double)
    jet_mass = np.empty(len(jets), dtype=np.double)
    jet_mass_trimmed = np.empty(len(jets), dtype=np.double)
    images_zoom_pt = np.empty((len(jets), 25, 25), dtype=np.double)
    for j, jet in enumerate(jets):
        size = 2 * 80.385 / jet.jets[0]['pT']
        images_zoom_pt[j] = preprocess(
            jet.subjets, jet.trimmed_constit, zoom_edges,
            zoom=max(1. / size, 1.))
        dr_pt[j] = size, jet.subjet_dr
        jet_mass[j] = jet.jets[0]['mass']
        jet_mass_trimmed[j] = jet.jets[1]['mass']
    images_zoom_pt = images_zoom_pt.mean(axis=0)

    dr_pt_trimmed = np.empty((len(jets), 2), dtype=np.double)
    images_zoom_pt_trimmed = np.empty((len(jets), 25, 25), dtype=np.double)
    for j, jet in enumerate(jets):
        size = 2 * 80.385 / jet.jets[1]['pT']
        images_zoom_pt_trimmed[j] = preprocess(
            jet.subjets, jet.trimmed_constit, zoom_edges,
            zoom=max(1. / size, 1.))
        dr_pt_trimmed[j] = size, jet.subjet_dr
    images_zoom_pt_trimmed = images_zoom_pt_trimmed.mean(axis=0)

    ax[0, i].set_title('{0:.1f};{1:.1f}'.format(1, R))
    # plot jet images
    plot_jet_image(ax[0, i], images, label_axes=False, show_colorbar=False)
    plot_jet_image(ax[1, i], images_zoom_pt, label_axes=False, show_colorbar=False)
    plot_jet_image(ax[2, i], images_zoom_pt_trimmed, label_axes=False, show_colorbar=False)

    dr_pt = dr_pt[dr_pt[:,1] > 0]
    corr_pt = np.corrcoef(dr_pt[:,0], dr_pt[:,1])
    ax[3, i].hist2d(dr_pt[:,0], dr_pt[:,1],
                    bins=(np.linspace(0, 1, 50), np.linspace(0, 1, 50)),
                    cmap='Reds')
    plt.setp(ax[3, i].get_xticklabels()[0], visible=False)
    plt.setp(ax[3, i].get_xticklabels()[-1], visible=False)
    if i == 0:
        plt.setp(ax[3, i].get_yticklabels()[0], visible=False)
        plt.setp(ax[3, i].get_yticklabels()[-1], visible=False)
        ax[3, i].set_ylabel(r'$2m_W/p_T$')
        ax[3, i].set_xlabel(r'$\Delta R$ subjets')
    else:
        ax[3, i].yaxis.set_ticklabels([])

    ax[3, i].text(0.05, 0.95, '{0:.2f}%'.format(corr_pt[0, 1] * 100.),
        verticalalignment='top', horizontalalignment='left',
        transform=ax[3, i].transAxes)

    dr_pt_trimmed = dr_pt_trimmed[dr_pt_trimmed[:,1] > 0]
    corr_pt_trimmed = np.corrcoef(dr_pt_trimmed[:,0], dr_pt_trimmed[:,1])
    ax[4, i].hist2d(dr_pt_trimmed[:,0], dr_pt_trimmed[:,1],
                    bins=(np.linspace(0, 1, 50), np.linspace(0, 1, 50)),
                    cmap='Reds')
    plt.setp(ax[4, i].get_xticklabels()[0], visible=False)
    plt.setp(ax[4, i].get_xticklabels()[-1], visible=False)
    if i == 0:
        plt.setp(ax[4, i].get_yticklabels()[0], visible=False)
        plt.setp(ax[4, i].get_yticklabels()[-1], visible=False)
        ax[4, i].set_ylabel(r'$2m_W/p_T$ trimmed')
        ax[4, i].set_xlabel(r'$\Delta R$ subjets')
    else:
        ax[4, i].yaxis.set_ticklabels([])

    ax[4, i].text(0.05, 0.95, 'C = {0:.2f}%'.format(corr_pt_trimmed[0, 1] * 100.),
        verticalalignment='top', horizontalalignment='left',
        transform=ax[4, i].transAxes)

    vals1, _, _ = ax[5, i].hist(jet_mass, bins=np.linspace(65, 95, 50), label='Mass',
                  histtype='stepfilled', facecolor='none', edgecolor='red', normed=1)
    vals2, _, _ = ax[5, i].hist(jet_mass_trimmed, bins=np.linspace(65, 95, 50), label='Trimmed Mass',
                  histtype='stepfilled', facecolor='none', edgecolor='black', normed=1)
    ax[5, i].set_ylim((0, 1.3 * max(np.max(vals1), np.max(vals2))))
    if i == 0:
        ax[5, i].set_xlabel(r'Jet Mass [GeV]')
        ax[5, i].set_ylabel('Normalized to Unity')
        ax[5, i].legend(frameon=False)

ax[0, 0].axes.get_yaxis().set_visible(True)
ax[1, 0].axes.get_yaxis().set_visible(True)
ax[2, 0].axes.get_yaxis().set_visible(True)
ax[0, 0].set_ylabel('No Zoom')
ax[1, 0].set_ylabel(r'Zoom with $p_T$')
ax[2, 0].set_ylabel(r'Zoom with trimmed $p_T$')

fig.tight_layout()
fig.savefig('show_clustering.png')
fig.savefig('show_clustering.pdf')
