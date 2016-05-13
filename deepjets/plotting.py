

def plot_kinematics(signal, background, nbins=100,
                    mass_range=(50., 110.), pt_range=(150., 550.),
                    linewidth=1, title=None):
    import numpy as np
    from matplotlib import pyplot as plt
    import h5py

    pt_min, pt_max = pt_range
    mass_min, mass_max = mass_range

    plt.style.use('seaborn-white')

    signal_h5file_events = h5py.File(signal, 'r')
    signal_aux = signal_h5file_events['auxvars']

    background_h5file_events = h5py.File(background, 'r')
    background_aux = background_h5file_events['auxvars']

    #signal_selection = (signal_aux['mass_trimmed'] > mass_min) & (signal_aux['mass_trimmed'] < mass_max)
    #background_selection = (background_aux['mass_trimmed'] > mass_min) & (background_aux['mass_trimmed'] < mass_max)

    #print signal_selection.sum()
    #print background_selection.sum()
    #print (signal_selection & ((signal_aux['pt_trimmed'] > 250) & (signal_aux['pt_trimmed'] < 300))).sum()
    #print (background_selection & ((background_aux['pt_trimmed'] > 250) & (background_aux['pt_trimmed'] < 300))).sum()

    signal_selection = slice(0, None)
    background_selection = slice(0, None)

    if 'weights' in signal_aux.dtype.names:
        signal_weights = signal_aux['weights']
    else:
        signal_weights = np.ones(len(signal_aux))

    if 'weights' in background_aux.dtype.names:
        background_weights = background_aux['weights']
    else:
        background_weights = np.ones(len(background_aux))

    signal_weights = signal_weights[signal_selection]
    background_weights = background_weights[background_selection]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    if title is not None:
        fig.suptitle(title, fontsize=16)

    vals1, _, _ = ax[0, 0].hist(signal_aux['pt_trimmed'][signal_selection],
        bins=np.linspace(pt_min, pt_max, nbins),
        histtype='stepfilled', facecolor='none', edgecolor='blue', normed=1,
        linewidth=linewidth,
        label=r'W jets', weights=signal_weights)
    vals2, _, _ = ax[0, 0].hist(background_aux['pt_trimmed'][background_selection],
        bins=np.linspace(pt_min, pt_max, nbins),
        histtype='stepfilled', facecolor='none', edgecolor='black', normed=1,
        linestyle='dotted', linewidth=linewidth,
        label='QCD jets', weights=background_weights)
    ax[0, 0].set_ylim((0, 1.3 * max(np.max(vals1), np.max(vals2))))
    ax[0, 0].set_ylabel('Normalized to Unity')
    ax[0, 0].set_xlabel(r'Trimmed $p_{T}$ [GeV]', fontsize=12)

    p1, = ax[0, 0].plot([0, 0], label='W jets', color='blue')
    p2, = ax[0, 0].plot([0, 0], label='QCD jets', color='black', linestyle='dotted')
    ax[0, 0].legend([p1, p2], ['W jets', 'QCD jets'], frameon=False, handlelength=3)
    ax[0, 0].set_xlim((pt_min, pt_max))
    ax[0, 0].ticklabel_format(style='sci', scilimits=(0,0), axis='y')

    vals1, _, _ = ax[0, 1].hist(signal_aux['mass_trimmed'][signal_selection],
        bins=np.linspace(mass_min - 10, mass_max + 10, nbins),
        histtype='stepfilled', facecolor='none', edgecolor='blue', normed=1,
        linewidth=linewidth,
        label=r'W jets', weights=signal_weights)
    vals2, _, _ = ax[0, 1].hist(background_aux['mass_trimmed'][background_selection],
        bins=np.linspace(mass_min - 10, mass_max + 10, nbins),
        histtype='stepfilled', facecolor='none', edgecolor='black', normed=1,
        linestyle='dotted', linewidth=linewidth,
        label='QCD jets', weights=background_weights)
    ax[0, 1].set_ylim((0, 1.3 * max(np.max(vals1), np.max(vals2))))
    ax[0, 1].set_ylabel('Normalized to Unity')
    ax[0, 1].set_xlabel(r'Trimmed Mass [GeV]', fontsize=12)

    p1, = ax[0, 1].plot([0, 0], label='W jets', color='blue')
    p2, = ax[0, 1].plot([0, 0], label='QCD jets', color='black', linestyle='dotted')
    ax[0, 1].legend([p1, p2], ['W jets', 'QCD jets'], frameon=False, handlelength=3)
    ax[0, 1].set_xlim((mass_min - 10, mass_max + 10))

    signal_tau21 = np.true_divide(signal_aux['tau_2'], signal_aux['tau_1'])[signal_selection]
    background_tau21 = np.true_divide(background_aux['tau_2'], background_aux['tau_1'])[background_selection]

    # remove NaN infinity and zero
    signal_tau21_nonan = ~np.isnan(signal_tau21) & ~np.isinf(signal_tau21) & (signal_tau21 != 0)
    background_tau21_nonan = ~np.isnan(background_tau21) & ~np.isinf(background_tau21) & (background_tau21 != 0)

    vals1, _, _ = ax[1, 0].hist(signal_tau21[signal_tau21_nonan],
        bins=np.linspace(0, 1, nbins),
        histtype='stepfilled', facecolor='none', edgecolor='blue', normed=1,
        linewidth=linewidth,
        label=r'W jets', weights=signal_weights[signal_tau21_nonan])
    vals2, _, _ = ax[1, 0].hist(background_tau21[background_tau21_nonan],
        bins=np.linspace(0, 1, nbins),
        histtype='stepfilled', facecolor='none', edgecolor='black', normed=1,
        linestyle='dotted', linewidth=linewidth,
        label='QCD jets', weights=background_weights[background_tau21_nonan])
    ax[1, 0].set_ylim((0, 1.3 * max(np.max(vals1), np.max(vals2))))
    ax[1, 0].set_ylabel('Normalized to Unity')
    ax[1, 0].set_xlabel(r'$\tau_{21}$', fontsize=12)

    p1, = ax[1, 0].plot([0, 0], label='W jets', color='blue')
    p2, = ax[1, 0].plot([0, 0], label='QCD jets', color='black', linestyle='dotted')
    ax[1, 0].legend([p1, p2], ['W jets', 'QCD jets'], frameon=False, handlelength=3)
    ax[1, 0].set_xlim((0, 1))

    vals1, _, _ = ax[1, 1].hist(signal_aux['subjet_dr'][signal_selection],
        bins=np.linspace(0, 1.2, nbins),
        histtype='stepfilled', facecolor='none', edgecolor='blue', normed=1,
        linewidth=linewidth,
        label=r'W jets', weights=signal_weights)
    vals2, _, _ = ax[1, 1].hist(background_aux['subjet_dr'][background_selection],
        bins=np.linspace(0, 1.2, nbins),
        histtype='stepfilled', facecolor='none', edgecolor='black', normed=1,
        linestyle='dotted', linewidth=linewidth,
        label='QCD jets', weights=background_weights)
    ax[1, 1].set_ylim((0, 1.3 * max(np.max(vals1), np.max(vals2))))
    ax[1, 1].set_ylabel('Normalized to Unity')
    ax[1, 1].set_xlabel(r'Subjets $\Delta R$', fontsize=12)

    p1, = ax[1, 1].plot([0, 0], label='W jets', color='blue')
    p2, = ax[1, 1].plot([0, 0], label='QCD jets', color='black', linestyle='dotted')
    ax[1, 1].legend([p1, p2], ['W jets', 'QCD jets'], frameon=False, handlelength=3)
    ax[1, 1].set_xlim((0, 1.2))

    fig.tight_layout()
    if title is not None:
        plt.subplots_adjust(top=0.93)
    return fig

