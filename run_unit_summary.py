from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmap
from loguru import logger

import spikeinterface.core as si
import spikeinterface.extractors as se

from pipeline.plotting import (
    plot_unit_positions,
    plot_unit_spikes,
    plot_unit_template,
    plot_template_shanks,
    plot_template_animation,
)


def run(probe_path: Path, skip_animation: bool = False):
    concat_path = probe_path / 'concat'
    ks_output = probe_path / 'kilosort'
    save_path = probe_path / 'unit_summary'
    save_path.mkdir(parents=True, exist_ok=True)

    # LOAD PROBE CONTACTS 
    ch_pos = np.load(ks_output / 'channel_positions.npy')
    ch_shanks = np.load(ks_output / 'channel_shanks.npy').astype(int)
    contacts_df = pd.DataFrame(ch_pos, columns=['x', 'y'])
    contacts_df['shank'] = ch_shanks.astype(int)

    # LOAD UNITS FROM KILOSORT OUTPUT
    unit_labels = pd.read_csv(ks_output / 'cluster_group.tsv', sep='\t', index_col=0)
    good_units = unit_labels[unit_labels['KSLabel'] == 'good'].index.values
    logger.info(f"Found {len(good_units)} good units")

    clu = np.load(ks_output / 'spike_clusters.npy', mmap_mode='r')
    spike_locations = np.load(ks_output / 'spike_positions.npy', mmap_mode='r')

    good_mask = np.isin(clu, good_units)
    spikes_df = pd.DataFrame(spike_locations[good_mask], columns=['x', 'y'])
    spikes_df['cluster'] = clu[good_mask]
    units_df = spikes_df.groupby('cluster').agg(['median', 'min', 'max'])
    units_df.columns = ['_'.join(col) for col in units_df.columns]

    ### SORTING ANALYZER
    fs = 30000.0

    processing_path = probe_path / 'processing'
    if processing_path.exists():
        logger.info("Loading postprocess from processing folder")
        pp = si.load_sorting_analyzer(probe_path / 'postprocess')
    else:
        ## Load recording and sorting objects
        rec = si.load(str(concat_path))
        sorting = se.read_kilosort(ks_output).select_units(good_units)

        pp = si.create_sorting_analyzer(
            sorting=sorting,
            recording=rec,
            sparse=False,
            folder=probe_path / 'postprocess',
            format='binary_folder',
            overwrite=False
        )

        job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)

        compute_dict = {
            'random_spikes': {'method': 'uniform', 'max_spikes_per_unit': 10000},
            'templates': {'operators': ["average", "std"]}
        }

        pp.compute(compute_dict, **job_kwargs)

    template_ext = pp.get_extension('templates')
    ms_before = template_ext.params['ms_before']
    ms_after = template_ext.params['ms_after']

    # Color scheme
    plt.style.use('dark_background')
    heatmap = cmap.Colormap('seaborn:vlag').to_matplotlib()
    distinct_colors = cmap.Colormap('tab20').to_matplotlib()

    plot_unit_positions(units_df, ch_pos, colormap=distinct_colors, save_path=probe_path)
    logger.success("Saved unit positions overview")

    # Per-unit plots
    for idx, cluster_id in enumerate(good_units):
        logger.info(f"[{idx + 1}/{len(good_units)}] Unit {cluster_id}")

        unit_template = template_ext.get_unit_template(cluster_id, operator='average')
        # Centre each channel to zero
        unit_template = unit_template - unit_template.mean(axis=0, keepdims=True)

        unit = units_df.loc[cluster_id]
        spikes = spikes_df[spikes_df['cluster'] == cluster_id]

        plot_unit_spikes(cluster_id, unit, spikes, contacts_df,
                         colormap=distinct_colors, save_path=save_path)

        plot_unit_template(cluster_id, unit_template, ms_before, ms_after,
                           cmap_heat=heatmap, save_path=save_path, fs=fs)

        plot_template_shanks(cluster_id, unit_template, contacts_df,
                             ms_before, ms_after,
                             cmap_heat=heatmap, save_path=save_path, fs=fs)

        if not skip_animation:
            plot_template_animation(cluster_id, unit_template, contacts_df,
                                    ms_before, ms_after,
                                    cmap_heat=heatmap, save_path=save_path, fs=fs)

        logger.success(f"Unit {cluster_id} done")

    logger.success("ALL UNITS COMPLETED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe_path', type=Path)
    parser.add_argument('--skip-animation', action='store_true')
    args = parser.parse_args()

    run(args.probe_path, skip_animation=args.skip_animation)
