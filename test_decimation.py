"""
Test script to verify DecimatedRecording logic with synthetic data.
"""
import numpy as np
import spikeinterface as si
from spikeinterface.core import BaseRecording, BaseRecordingSegment


class SimpleArrayRecording(BaseRecording):
    """Mock recording backed by a numpy array for testing."""

    def __init__(self, traces_array, sampling_frequency):
        """
        Parameters
        ----------
        traces_array : np.ndarray
            Shape: (num_samples, num_channels)
        sampling_frequency : float
            Sampling rate in Hz
        """
        self._traces = traces_array
        num_samples, num_channels = traces_array.shape

        BaseRecording.__init__(
            self,
            sampling_frequency=sampling_frequency,
            channel_ids=list(range(num_channels)),
            dtype=traces_array.dtype
        )

        self.add_recording_segment(
            SimpleArrayRecordingSegment(traces_array, sampling_frequency)
        )


class SimpleArrayRecordingSegment(BaseRecordingSegment):
    """Segment backed by numpy array."""

    def __init__(self, traces_array, sampling_frequency):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._traces = traces_array
        self._num_samples = traces_array.shape[0]

    def get_num_samples(self):
        return self._num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        return self._traces[start_frame:end_frame, channel_indices]


class DecimatedRecording(BaseRecording):
    """Recording that decimates on-the-fly by selecting every k-th sample."""

    def __init__(self, parent_recording, decimation_factor):
        self._parent_recording = parent_recording
        self._decimation_factor = int(decimation_factor)

        # Update sampling frequency
        parent_fs = parent_recording.get_sampling_frequency()
        new_fs = parent_fs / self._decimation_factor

        BaseRecording.__init__(
            self,
            sampling_frequency=new_fs,
            channel_ids=parent_recording.get_channel_ids(),
            dtype=parent_recording.get_dtype()
        )

        # Copy properties
        self.copy_metadata(parent_recording)

        # Add segments
        for segment in parent_recording._recording_segments:
            self.add_recording_segment(
                DecimatedRecordingSegment(segment, self._decimation_factor)
            )


class DecimatedRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_segment, decimation_factor):
        self._parent_segment = parent_segment
        self._decimation_factor = decimation_factor

        # Calculate decimated length
        parent_samples = parent_segment.get_num_samples()
        decimated_samples = parent_samples // decimation_factor

        BaseRecordingSegment.__init__(
            self,
            sampling_frequency=parent_segment.sampling_frequency / decimation_factor
        )
        self._num_samples = decimated_samples

    def get_num_samples(self):
        return self._num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        # Map to parent frames
        parent_start = start_frame * self._decimation_factor
        parent_end = end_frame * self._decimation_factor

        parent_traces = self._parent_segment.get_traces(
            start_frame=parent_start,
            end_frame=parent_end,
            channel_indices=channel_indices
        )

        return parent_traces[::self._decimation_factor, :]


def test_basic_decimation():
    """Test that decimation selects correct samples."""
    print("\n=== Test 1: Basic Decimation ===")

    # Create simple test data: channel values are sample indices
    num_samples = 30
    num_channels = 2
    traces = np.arange(num_samples).reshape(-1, 1).repeat(num_channels, axis=1)
    # traces[:, 0] = [0, 1, 2, 3, ..., 29]
    # traces[:, 1] = [0, 1, 2, 3, ..., 29]

    print(f"Original traces shape: {traces.shape}")
    print(f"Original traces (channel 0): {traces[:, 0]}")

    # Create recording
    rec = SimpleArrayRecording(traces, sampling_frequency=1000.0)

    # Decimate by factor of 3
    decimation_factor = 3
    dec_rec = DecimatedRecording(rec, decimation_factor)

    print(f"\nDecimation factor: {decimation_factor}")
    print(f"Decimated num_samples: {dec_rec.get_num_frames()}")
    print(f"Expected: {num_samples // decimation_factor}")

    # Get all decimated traces
    dec_traces = dec_rec.get_traces(start_frame=0, end_frame=dec_rec.get_num_frames())
    print(f"Decimated traces (channel 0): {dec_traces[:, 0]}")

    # Expected: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    expected = traces[::decimation_factor, :]
    print(f"Expected (channel 0): {expected[:, 0]}")

    assert np.array_equal(dec_traces, expected), "Decimation mismatch!"
    print("✓ Basic decimation correct")


def test_chunked_decimation():
    """Test that chunked reading produces same result as full decimation."""
    print("\n=== Test 2: Chunked Decimation (Simulating Parallel Workers) ===")

    # Create test data
    num_samples = 24000  # 24000 samples
    num_channels = 4

    # Create identifiable pattern: sample_index * 10 + channel_index
    traces = np.zeros((num_samples, num_channels), dtype='float32')
    for ch in range(num_channels):
        traces[:, ch] = np.arange(num_samples) * 10 + ch

    print(f"Original traces shape: {traces.shape}")
    print(f"Original samples [0:12, ch0]: {traces[0:12, 0]}")

    # Create recording and decimate
    rec = SimpleArrayRecording(traces, sampling_frequency=30000.0)
    decimation_factor = 24
    dec_rec = DecimatedRecording(rec, decimation_factor)

    decimated_samples = dec_rec.get_num_frames()
    print(f"\nDecimated num_samples: {decimated_samples} (expected: {num_samples // decimation_factor})")

    # Get full decimated result (ground truth)
    full_decimated = dec_rec.get_traces(start_frame=0, end_frame=decimated_samples)
    print(f"Full decimated shape: {full_decimated.shape}")
    print(f"Full decimated [0:10, ch0]: {full_decimated[0:10, 0]}")

    # Simulate 3 workers processing chunks
    chunk_size = decimated_samples // 3  # Split into 3 chunks

    print(f"\n--- Simulating {3} parallel workers ---")
    print(f"Chunk size (in decimated space): {chunk_size}")

    chunks = []
    for worker_id in range(3):
        start = worker_id * chunk_size
        end = start + chunk_size if worker_id < 2 else decimated_samples  # Last chunk gets remainder

        print(f"\nWorker {worker_id}: decimated frames [{start} → {end}]")
        print(f"  → Fetching from parent frames [{start * decimation_factor} → {end * decimation_factor}]")

        chunk = dec_rec.get_traces(start_frame=start, end_frame=end)
        print(f"  → Got chunk shape: {chunk.shape}")
        print(f"  → First 5 values (ch0): {chunk[0:5, 0]}")

        chunks.append(chunk)

    # Concatenate chunks (simulating what would be written to file)
    reconstructed = np.concatenate(chunks, axis=0)

    print(f"\n--- Verification ---")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Full decimated shape: {full_decimated.shape}")

    # Verify they match
    assert reconstructed.shape == full_decimated.shape, "Shape mismatch!"
    assert np.array_equal(reconstructed, full_decimated), "Reconstruction doesn't match full decimation!"

    print("✓ Chunked decimation matches full decimation")
    print("✓ Parallel workers would produce correct output")


def test_alignment():
    """Test that chunk boundaries are properly aligned."""
    print("\n=== Test 3: Alignment Verification ===")

    # Simple sequential data
    num_samples = 100
    num_channels = 1
    traces = np.arange(num_samples).reshape(-1, 1)

    rec = SimpleArrayRecording(traces, sampling_frequency=1000.0)
    decimation_factor = 5
    dec_rec = DecimatedRecording(rec, decimation_factor)

    # Get two adjacent chunks
    chunk1 = dec_rec.get_traces(start_frame=0, end_frame=10)  # decimated frames 0-10
    chunk2 = dec_rec.get_traces(start_frame=10, end_frame=20)  # decimated frames 10-20

    print(f"Chunk 1 (decimated 0→10): {chunk1.flatten()}")
    print(f"Chunk 2 (decimated 10→20): {chunk2.flatten()}")

    # Verify no gaps or overlaps
    assert chunk1[-1, 0] == 45, f"Chunk 1 last value should be 45, got {chunk1[-1, 0]}"
    assert chunk2[0, 0] == 50, f"Chunk 2 first value should be 50, got {chunk2[0, 0]}"

    print("✓ No gaps between chunks")
    print("✓ Boundaries properly aligned")


def test_edge_cases():
    """Test edge cases with small arrays."""
    print("\n=== Test 4: Edge Cases ===")

    # Very small array
    traces = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
    rec = SimpleArrayRecording(traces, sampling_frequency=100.0)

    dec_rec = DecimatedRecording(rec, decimation_factor=3)
    result = dec_rec.get_traces(start_frame=0, end_frame=dec_rec.get_num_frames())

    expected = np.array([[0], [3], [6]])
    print(f"Small array result: {result.flatten()}")
    print(f"Expected: {expected.flatten()}")

    assert np.array_equal(result, expected), "Small array decimation failed!"
    print("✓ Small array decimation correct")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DecimatedRecording Implementation")
    print("=" * 60)

    test_basic_decimation()
    test_chunked_decimation()
    test_alignment()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nConclusion:")
    print("• Decimation correctly selects every k-th sample")
    print("• Parallel chunks can be processed independently")
    print("• No interference between workers")
    print("• Chunk boundaries are properly aligned")
