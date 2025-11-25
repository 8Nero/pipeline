import spikeinterface as si
import spikeinterface.extractors as se

import numpy as np
from pipeline.utils import log_recording
from pathlib import Path
from loguru import logger

from pipeline.utils import load_config

class ProbeRun:
    """ Represents a single probe in an OpenEphys session."""
    
    
    def __init__(self, run_path, name, stream_id):
        self.run_path       = run_path
        self.name           = name
        self.stream_id      = stream_id
        self._recording     = None

    def __repr__(self):
        return f"ProbeRun(name={self.name}, stream_id={self.stream_id})"

    @property
    def recording(self):
        if self._recording is None:
            self._recording = se.read_openephys(self.run_path, stream_id=self.stream_id)
        return self._recording
    
class MultiProbeRun:
    """ Represents multiple probes in a single OpenEphys session ."""


    def __init__(self, run_path):
        logger.info(f"Initializing MultiProbeRun for path: {run_path}")
        self.run_path = run_path
        self.session_name = Path(run_path).name
        self._recording = None
        self._probes = {}

    def load_recordings(self):
        """ Load recordings for all probes matching the filter."""
        logger.info(f"Loading recordings from {self.run_path}")
        stream_names, stream_ids = se.get_neo_streams('openephysbinary', self.run_path)

        for stream_name, stream_id in zip(stream_names, stream_ids):
            logger.info(f"Found stream: {stream_name} with ID: {stream_id}")
            # Extract probe name (e.g., "OneBox-0.ProbeA" -> "ProbeA")
            probe_name = stream_name.split(".")[-1]
            if "SYNC" not in probe_name:
                probe = ProbeRun(self.run_path, stream_name, stream_id)
                self._probes[probe_name] = probe

    def concatenate(self, probe_name):
        """ Concatenate probe across recordings."""
        if probe_name not in self._probes:
            raise ValueError(f"Probe {probe_name} not found in this session.")
        recording = self._probes[probe_name].recording
        log_recording(recording, f"Probe_{probe_name}_Recording")
        return self._probes[probe_name].recording

    def __getitem__(self, probe_name):
        return self._probes[probe_name]

    def __repr__(self):
        return f"MultiProbeRun(run_path={self.run_path}, probes={list(self._probes.keys())})"

class Experiment:
    """ Represents an experiment with multiple sessions."""


    def __init__(self, recording_paths):
        self.sessions = [MultiProbeRun(path) for path in recording_paths]
        self._probes = {}

    def preprocess(self):
        """ Preprocess all sessions."""
        for session in self.sessions:
            logger.info(f"Preprocessing session at {session.run_path}")
            session.load_recordings()

    def concatenate(self, probe_filter, save_path=None):
        """ Concatenate recordings for each probe across sessions."""
        for probe_name in probe_filter:
            logger.info(f"Concatenating sessions for probe: {probe_name}")
            probe_recordings = [session[probe_name].recording for session in self.sessions]
            concatenated_recording = si.concatenate_recordings(probe_recordings)
            log_recording(concatenated_recording, f"Concatenated_Probe_{probe_name}")
            self._probes[probe_name] = concatenated_recording
        
        if save_path:
            for probe_name, recording in self._probes.items():
                si.write_binary_recording(recording, save_path / f"Concatenated_Probe_{probe_name}")
        return self._probes
    
    def __repr__(self):
        return f"Experiment(sessions={len(self.sessions)})"


if __name__ == "__main__":
    conf_path = Path("R:\\Basic_Sciences\\Phys\\SenzaiLab\\pipeline_output\\configs\\config_mouse3_hunt-sleepx2.yaml")
    config = load_config(conf_path)

    exp = Experiment(recording_paths=config['recording_paths'])
    exp.preprocess()
    
    probe = exp.concatenate(probe_filter=['ProbeB'])['ProbeB']
    probe = probe.frame_slice(start_frame=0, end_frame=8 * 60 * 60 * 30000)
    
    save_path = Path(r"E:\pipeline_output\mouse3_hunt-sleepx2\ProbeB\concat")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    probe.save(format="binary", folder=save_path, n_jobs=4, chunk_duration='2s', mp_context='spawn')
