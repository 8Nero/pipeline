from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize, Colormap

try:
    import imageio_ffmpeg
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass
def plot_unit_positions(units_df: pd.DataFrame, ch_pos: np.ndarray,
                       colormap: Colormap | None = None,
                       save_path: Path | None = None):
    """Scatter plot of all good-unit median positions on the probe layout."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(ch_pos[:, 0], ch_pos[:, 1], s=25, color='white', marker='s', alpha=0.3)

    for cluster_id, unit in units_df.iterrows():
        ax.scatter(unit['x_median'], unit['y_median'], s=10, color=colormap(cluster_id % 20), marker='o')

    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_xmargin(0.15)

    if save_path is not None:
        ax.set_title(f'{save_path.name}: Unit Positions')
        fig.savefig(save_path / 'unit_positions.png', dpi=320)
        plt.close(fig)
    else:
        return fig, ax


def plot_unit_spikes(cluster_id: int, unit: pd.Series, spikes: pd.DataFrame,
                     contacts_df: pd.DataFrame,
                     colormap: Colormap | None = None,
                     save_path: Path | None = None,
                     x_margin: float = 18, y_margin: float = 45):
    """Spike cloud for a single unit with nearby contact annotations."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(spikes['x'], spikes['y'], s=1, color=colormap((cluster_id + 1) % 20), marker='.')
    ax.scatter(unit['x_median'], unit['y_median'], s=10, color=colormap(cluster_id % 20), marker='o')

    x_mask = (contacts_df['x'] > (unit['x_min'] - x_margin)) & (contacts_df['x'] < (unit['x_max'] + x_margin))
    y_mask = (contacts_df['y'] > (unit['y_min'] - y_margin)) & (contacts_df['y'] < (unit['y_max'] + y_margin))
    for contact_id, contact in contacts_df[x_mask & y_mask].iterrows():
        ax.annotate(str(contact_id),
                    xy=(contact['x'], contact['y']),
                    fontsize=12, ha='center', va='center',
                    color='white', alpha=1,
                    bbox=dict(boxstyle='round,pad=0.3', fc='gray', alpha=0.5)
                    )

    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_xlim(unit['x_min'] - x_margin, unit['x_max'] + x_margin)
    ax.set_ylim(unit['y_min'] - y_margin, unit['y_max'] + y_margin)
    ax.set_title(f'Unit {cluster_id} Spike Locations')
    if save_path is not None:
        fig.savefig(save_path / f'unit_{cluster_id}_spikes.png', dpi=320)
        plt.close(fig)
    else:
        return fig, ax


def plot_unit_template(cluster_id: int,unit_template: np.ndarray,
                       ms_before: float, ms_after: float,
                       cmap_heat: Colormap | None = None,
                       save_path: Path | None = None, fs: float = 30000.0):
    """Heatmap and line overlay of a unit's template across all channels."""
    time_ms = np.arange(-ms_before, ms_after, 1000 / fs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.25]})
    fig.suptitle(f'Unit {cluster_id} — All Channels')

    ## All channels Heatmap
    im = axes[0].imshow(unit_template.T,
                    aspect='auto', cmap=cmap_heat,
                    origin='lower', interpolation='none',
                    extent=[time_ms[0], time_ms[-1], 0, unit_template.shape[1]]
                    )

    axes[0].set_xlabel('Time (ms)')
    axes[0].set_xticks(np.linspace(-ms_before, ms_after, 4))
    axes[0].set_ylabel('Channel')

    fig.colorbar(im, ax=axes[0])

    ## All channels line plot
    for ch_id in range(unit_template.shape[1]):
        axes[1].plot(time_ms, unit_template[:, ch_id])

    axes[1].set_xlabel('Time (ms)')
    axes[1].set_xticks(np.linspace(-ms_before, ms_after, 4))
    axes[1].set_title(f'Unit {cluster_id}')

    if save_path is not None:
        fig.savefig(save_path / f'unit_{cluster_id}_template.png', dpi=320)
        plt.close(fig)
    else:
        return fig, axes


def plot_template_shanks(cluster_id: int, unit_template: np.ndarray,
                         contacts_df: pd.DataFrame,
                         ms_before: float, ms_after: float,
                         cmap_heat: Colormap | None = None,
                         save_path: Path | None = None, fs: float = 30000.0):
    """Per-shank heatmaps of a unit's template along the depth axis."""
    ch_pos = contacts_df[['x', 'y']].values
    ch_shanks = contacts_df['shank'].values
    time_ms = np.arange(-ms_before, ms_after, 1000 / fs)
    shanks = np.unique(ch_shanks)

    fig, axes = plt.subplots(2, len(shanks),
                        figsize=(16, 5),
                        sharex=False, sharey=False,
                        gridspec_kw={'height_ratios': [1, 0.05], 'wspace': 0.4, 'hspace': 0.4}
                        )

    if len(shanks) == 1:
        axes = axes.reshape(2, 1)

    # Derive y-tick step and range from all channel depths
    all_depths = ch_pos[:, 1]
    depth_step = 200
    tick_start = int(np.ceil(all_depths.min() / depth_step) * depth_step)
    tick_end = int(np.floor(all_depths.max() / depth_step) * depth_step)
    depth_ticks = np.arange(tick_start, tick_end + 1, depth_step)

    fig.suptitle(f'Unit {cluster_id}')
    for i, shank_id in enumerate(shanks):
        mask = ch_shanks == shank_id
        ch_indices = mask.nonzero()[0]

        # Depth-sort channels
        depth_order = np.argsort(ch_pos[ch_indices, 1])
        sorted_indices = ch_indices[depth_order]
        depths = ch_pos[sorted_indices, 1]

        # Template heatmap
        im = axes[0, i].imshow(unit_template[:, sorted_indices].T,
                            aspect='auto', cmap=cmap_heat,
                            origin='lower', interpolation='none',
                            extent=[time_ms[0], time_ms[-1], depths[0], depths[-1]]
                            )

        # X ticks
        axes[0, i].set_xlabel('Time (ms)')
        axes[0, i].set_xticks(np.linspace(-ms_before, ms_after, 4))

        # Y ticks (µm)
        shank_ticks = depth_ticks[(depth_ticks >= depths[0]) & (depth_ticks <= depths[-1])]
        axes[0, i].set_yticks(shank_ticks)
        axes[0, i].set_title(f'Shank {shank_id}')

        # Per-shank colorbar in the bottom row
        fig.colorbar(im, cax=axes[1, i], orientation='horizontal')

    axes[0, 0].set_ylabel('Depth (µm)')
    if save_path is not None:
        fig.savefig(save_path / f'unit_{cluster_id}_template_shanks.png', dpi=320)
        plt.close(fig)
    else:
        return fig, axes


def plot_shankwise_waveforms(cluster_id: int, waveforms: np.ndarray,
                             unit_template: np.ndarray,
                             contacts_df: pd.DataFrame,
                             ms_before: float, ms_after: float,
                             n_display: int = 200,
                             save_path: Path | None = None, fs: float = 30000.0):
    """Per-shank waveform overlay and peak-to-peak histogram on the best channel.

    Top row: undersampled individual waveforms (thin, transparent) with mean overlay.
    Bottom row: peak-to-peak amplitude distribution across all extracted waveforms.

    Parameters
    ----------
    waveforms : (n_spikes, n_samples, n_channels) array
        Raw waveform snippets extracted from the recording.
    unit_template : (n_samples, n_channels) array
        Zero-centred mean template, used to find the peak channel per shank.
    """
    ch_shanks = contacts_df['shank'].values
    shanks = np.unique(ch_shanks)
    time_ms = np.arange(-ms_before, ms_after, 1000 / fs)
    n_samples = min(waveforms.shape[1], len(time_ms))
    n_total = waveforms.shape[0]

    fig, axes = plt.subplots(2, len(shanks),
                             figsize=(5 * len(shanks), 8),
                             gridspec_kw={'height_ratios': [2, 1]},
                             squeeze=False)
    fig.suptitle(f'Unit {cluster_id} — Per-Shank Waveforms ({n_total} extracted)')

    if n_total > n_display:
        rng = np.random.default_rng(42)
        display_idx = rng.choice(n_total, n_display, replace=False)
    else:
        display_idx = np.arange(n_total)

    for i, sid in enumerate(shanks):
        mask = ch_shanks == sid
        ch_indices = mask.nonzero()[0]

        # Peak channel per shank from mean template
        shank_template = unit_template[:, ch_indices]
        ptp_mean = shank_template.max(axis=0) - shank_template.min(axis=0)
        best_local = ptp_mean.argmax()
        best_ch = ch_indices[best_local]
        mean_ptp_val = ptp_mean[best_local]

        # Waveform overlay
        ax_wf = axes[0, i]
        for idx in display_idx:
            ax_wf.plot(time_ms[:n_samples], waveforms[idx, :n_samples, best_ch],
                       color='C0', alpha=0.05, linewidth=0.5)
        ax_wf.plot(time_ms[:n_samples], unit_template[:n_samples, best_ch],
                   color='yellow', linewidth=1.5, label='mean')
        ax_wf.set_title(f'Shank {sid} — Ch {best_ch}\nmean p2p = {mean_ptp_val:.1f} µV')
        ax_wf.set_xlabel('Time (ms)')
        ax_wf.set_ylabel('µV')
        ax_wf.legend(fontsize=8)

        # Amplitude histogram
        ax_hist = axes[1, i]
        ch_wfs = waveforms[:, :n_samples, best_ch]
        all_ptp = ch_wfs.max(axis=1) - ch_wfs.min(axis=1)
        median_ptp = np.median(all_ptp)
        ax_hist.hist(all_ptp, bins=50, color='C0', alpha=0.7,
                     edgecolor='white', linewidth=0.3)
        ax_hist.axvline(median_ptp, color='cyan', linestyle='-',
                        linewidth=1, label=f'median = {median_ptp:.1f}')
        ax_hist.axvline(mean_ptp_val, color='yellow', linestyle='--',
                        linewidth=1, label=f'template = {mean_ptp_val:.1f}')
        ax_hist.set_xlabel('Peak-to-peak (µV)')
        ax_hist.set_ylabel('Count')
        ax_hist.legend(fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path / f'unit_{cluster_id}_shankwise_waveforms.png', dpi=320)
        plt.close(fig)
    else:
        return fig, axes


def plot_template_animation(cluster_id: int, unit_template: np.ndarray,
                            contacts_df: pd.DataFrame,
                            ms_before: float, ms_after: float,
                            cmap_heat: Colormap | None = None,
                            save_path: Path | None = None, fs: float = 30000.0):
    """Animated scatter of template waveform propagation across shanks."""
    ch_pos = contacts_df[['x', 'y']].values
    ch_shanks = contacts_df['shank'].values
    time_ms = np.arange(-ms_before, ms_after, 1000 / fs)
    shanks = np.unique(ch_shanks)
    # Per-shank color normalization
    shank_norms = {}
    for sid in shanks:
        mask = ch_shanks == sid
        shank_data = unit_template[:, mask]
        shank_norms[sid] = Normalize(vmin=shank_data.min(), vmax=shank_data.max())

    fig, axes = plt.subplots(1, len(shanks), figsize=(2.5 * len(shanks), 10), sharey=True)
    if len(shanks) == 1:
        axes = [axes]

    scatters = {}
    for i, sid in enumerate(shanks):
        ax = axes[i]
        mask = ch_shanks == sid
        pos = ch_pos[mask]

        colors = shank_norms[sid](unit_template[0, mask])
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=colors, cmap=cmap_heat,
                        vmin=0, vmax=1, s=80, marker='s', edgecolors='gray',
                        linewidths=0.9)
        scatters[sid] = (sc, mask)

        ax.set_title(f'Shank {sid}')

        ax.set_xticks([])
        ax.set_xlabel('')
        ax.tick_params(bottom=False)
        ax.margins(x=2, y=0.05)

    axes[0].set_ylabel('y (µm)')

    fig.subplots_adjust(wspace=0.5)
    time_text = fig.suptitle(f't = {time_ms[0]:.2f} ms', fontsize=14)

    def update(frame):
        artists = [time_text]
        for sid in shanks:
            sc, mask = scatters[sid]
            colors = shank_norms[sid](unit_template[frame, mask])
            sc.set_array(colors)
            artists.append(sc)
        time_text.set_text(f't = {time_ms[frame]:.2f} ms')
        return artists

    anim = FuncAnimation(fig, update, frames=np.arange(len(time_ms)), interval=50, blit=False)
    if save_path is not None:
        anim.save(save_path / f'unit_{cluster_id}_template_shanks.mp4', fps=20, dpi=320)
        plt.close(fig)
    else:
        from IPython.display import HTML
        return HTML(anim.to_jshtml())
