"""
Decimation classes for downsampling SpikeInterface recordings.

Designed for parallelized EEG extraction from high-frequency recordings
using write_binary_recording.
"""
import spikeinterface as si
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from pathlib import Path
from loguru import logger


class DecimatedRecordingSegment(BaseRecordingSegment):
    """Segment that decimates parent segment on-the-fly."""
    
    def __init__(self, parent_segment, decimation_factor):
        self._parent_segment = parent_segment
        self._decimation_factor = int(decimation_factor)
        BaseRecordingSegment.__init__(self, **parent_segment.get_times_kwargs())
    
    def get_num_samples(self):
        parent_samples = self._parent_segment.get_num_samples()
        return parent_samples // self._decimation_factor
    
    def get_traces(self, start_frame, end_frame, channel_indices):
        # Map decimated frames back to parent frames
        parent_start = start_frame * self._decimation_factor
        parent_end = end_frame * self._decimation_factor
        
        # Get parent traces
        parent_traces = self._parent_segment.get_traces(
            parent_start, parent_end, channel_indices
        )
        
        # Decimate by selecting every k-th sample
        return parent_traces[::self._decimation_factor, :]


class DecimatedRecording(BaseRecording):
    """
    Recording that decimates on-the-fly by selecting every k-th sample.
    
    Designed for use with write_binary_recording for parallelized downsampling.
    Does not copy metadata - only preserves channel IDs, sampling frequency, and dtype.
    
    Parameters
    ----------
    parent_recording : BaseRecording
        The recording to decimate (e.g., BinaryFolderRecording at 30kHz)
    decimation_factor : int
        Decimation factor (e.g., 24 for 30kHz -> 1.25kHz)
    
    Example
    -------
    >>> # Load concatenated binary recording
    >>> concat_rec = si.load('path/to/concat')
    >>> # Decimate 30kHz -> 1.25kHz
    >>> eeg_rec = DecimatedRecording(concat_rec, decimation_factor=24)
    >>> # Save with parallelization
    >>> eeg_rec.save(format='binary', folder='eeg', n_jobs=8, chunk_duration='1s')
    
    Notes
    -----
    - Works with single or multi-segment recordings
    - No anti-aliasing filter applied (simple decimation)
    - Preserves dtype from parent recording
    - Does NOT copy probe geometry or other metadata
    """

    def __init__(self, parent_recording, decimation_factor):
        self._parent_recording = parent_recording
        self._decimation_factor = int(decimation_factor)

        parent_fs = parent_recording.get_sampling_frequency()
        new_fs = parent_fs / self._decimation_factor

        # Initialize WITHOUT copying metadata (no probe geometry, etc.)
        BaseRecording.__init__(
            self,
            sampling_frequency=new_fs,
            channel_ids=parent_recording.get_channel_ids(),
            dtype=parent_recording.get_dtype()
        )

        # Register decimated segments for each parent segment
        for parent_segment in parent_recording._recording_segments:
            decimated_segment = DecimatedRecordingSegment(parent_segment, self._decimation_factor)
            self.add_recording_segment(decimated_segment)
        
        # Don't copy any metadata - keep it clean for EEG output
        # If you need gain/offset annotations, you can uncomment this:
        # self.copy_metadata(parent_recording, only_main=True)


def downsample_eeg(
    eeg_folder: Path, 
    rec: si.BaseRecording, 
    target_fs: int = 1250,
    n_jobs: int = 8,
    chunk_duration: str = '1s'
) -> si.BaseRecording:
    """
    Decimate high-frequency recording to target sampling rate using parallel workers.
    
    This function replaces the old sequential downsampling approach with a parallelized
    version that uses write_binary_recording for much faster processing.
    
    Parameters
    ----------
    eeg_folder : Path
        Output directory for EEG data (will create eeg_data.dat inside)
    rec : si.BaseRecording
        Concatenated BinaryFolderRecording (typically 30kHz, 384 channels)
    target_fs : int
        Target sampling frequency in Hz (default: 1250)
    n_jobs : int
        Number of parallel workers for write_binary_recording (default: 8)
    chunk_duration : str or float
        Chunk size for parallel processing (e.g., '1s' or 1.0) (default: '1s')
    
    Returns
    -------
    si.BaseRecording
        The saved decimated recording (BinaryFolderRecording)
    
    Example
    -------
    >>> from pathlib import Path
    >>> import spikeinterface as si
    >>> from pipeline.decimation import downsample_eeg
    >>> 
    >>> # Load concatenated recording
    >>> concat_rec = si.load('output/ProbeA/concat')
    >>> 
    >>> # Downsample with 8 parallel workers
    >>> eeg_rec = downsample_eeg(
    ...     eeg_folder=Path('output/ProbeA'),
    ...     rec=concat_rec,
    ...     target_fs=1250,
    ...     n_jobs=8
    ... )
    
    Notes
    -----
    - Output file: {eeg_folder}/eeg_data.dat (raw binary int16)
    - No anti-aliasing filter applied
    - Skips if eeg_data.dat already exists
    - Uses all available CPU cores by default
    """
    eeg_file = eeg_folder / 'eeg_data.dat'
    if eeg_file.exists():
        logger.info(f"EEG data already exists at {eeg_file}, skipping")
        return si.load(eeg_folder / 'eeg')
    
    fs = rec.get_sampling_frequency()
    decimation_factor = int(fs / target_fs)
    
    logger.info(f"Downsampling EEG to: {eeg_folder}")
    logger.info(f"Decimation factor: {decimation_factor}")
    logger.info(f"Input: {fs:.0f} Hz -> Output: {fs/decimation_factor:.2f} Hz")
    
    # Create decimated recording (no metadata, just data)
    dec_rec = DecimatedRecording(rec, decimation_factor)
    
    # Save with parallelization
    logger.info(f"Saving with {n_jobs} parallel workers...")
    saved_rec = dec_rec.save(
        format='binary',
        folder=eeg_folder / 'eeg',
        name='eeg_data.dat',
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
        overwrite=True
    )
    
    logger.success(f"EEG data saved: {eeg_file}")
    return saved_rec
