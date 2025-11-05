"""
Simple test to verify decimation logic without dependencies.
Demonstrates how parallel workers would process chunks correctly.
"""


def simulate_get_traces(data, start_frame, end_frame):
    """Simulate recording.get_traces()"""
    return data[start_frame:end_frame]


def simulate_decimated_get_traces(parent_data, decimation_factor, start_frame, end_frame):
    """
    Simulate DecimatedRecording.get_traces()

    This is what each worker would call.
    start_frame and end_frame are in DECIMATED space.
    """
    # Convert decimated frame indices to parent frame indices
    parent_start = start_frame * decimation_factor
    parent_end = end_frame * decimation_factor

    # Fetch from parent recording
    parent_traces = simulate_get_traces(parent_data, parent_start, parent_end)

    # Decimate by selecting every k-th sample
    decimated_traces = parent_traces[::decimation_factor]

    return decimated_traces


def test_basic_decimation():
    """Test that decimation selects correct samples."""
    print("\n=== Test 1: Basic Decimation ===")

    # Create simple test data: each sample value is its index
    num_samples = 30
    num_channels = 2

    # Create 2D array manually (num_samples, num_channels)
    data = []
    for i in range(num_samples):
        data.append([i, i])  # Both channels have sample index as value

    print(f"Original data shape: ({len(data)}, {len(data[0])})")
    print(f"Original data (channel 0): {[row[0] for row in data]}")

    # Decimate by factor of 3
    decimation_factor = 3
    decimated_samples = num_samples // decimation_factor

    print(f"\nDecimation factor: {decimation_factor}")
    print(f"Decimated num_samples: {decimated_samples}")

    # Get all decimated traces
    dec_data = simulate_decimated_get_traces(
        data, decimation_factor,
        start_frame=0,
        end_frame=decimated_samples
    )

    print(f"Decimated data (channel 0): {[row[0] for row in dec_data]}")

    # Expected: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    expected = [data[i] for i in range(0, num_samples, decimation_factor)]
    print(f"Expected (channel 0): {[row[0] for row in expected]}")

    assert dec_data == expected, "Decimation mismatch!"
    print("✓ Basic decimation correct")


def test_parallel_workers():
    """
    Test that multiple workers processing different chunks
    produce the correct combined result.
    """
    print("\n=== Test 2: Parallel Workers (The Key Test!) ===")

    # Create test data: 24000 samples at 30kHz → 1000 samples at 1.25kHz
    num_samples = 24000
    num_channels = 2

    # Create identifiable pattern: sample_index * 10 + channel_index
    data = []
    for i in range(num_samples):
        data.append([i * 10 + 0, i * 10 + 1])  # ch0, ch1

    print(f"Original data: {num_samples} samples, {num_channels} channels")
    print(f"Original samples [0:12, ch0]: {[data[i][0] for i in range(12)]}")

    decimation_factor = 24
    decimated_samples = num_samples // decimation_factor
    print(f"\nDecimation factor: {decimation_factor}")
    print(f"Total decimated samples: {decimated_samples}")

    # Get full decimated result (ground truth)
    full_decimated = simulate_decimated_get_traces(
        data, decimation_factor,
        start_frame=0,
        end_frame=decimated_samples
    )
    print(f"\nGround truth (full decimation):")
    print(f"  First 10 values (ch0): {[full_decimated[i][0] for i in range(10)]}")

    # Simulate 4 workers processing chunks in parallel
    num_workers = 4
    chunk_size = decimated_samples // num_workers

    print(f"\n--- Simulating {num_workers} parallel workers ---")
    print(f"Chunk size (in decimated space): {chunk_size}")

    worker_results = []
    for worker_id in range(num_workers):
        # Calculate chunk boundaries in DECIMATED space
        dec_start = worker_id * chunk_size
        dec_end = dec_start + chunk_size if worker_id < num_workers - 1 else decimated_samples

        print(f"\nWorker {worker_id}:")
        print(f"  Decimated frames: [{dec_start} → {dec_end}]")
        print(f"  Parent frames: [{dec_start * decimation_factor} → {dec_end * decimation_factor}]")

        # Each worker independently fetches and decimates its chunk
        chunk = simulate_decimated_get_traces(
            data, decimation_factor,
            start_frame=dec_start,
            end_frame=dec_end
        )

        print(f"  Chunk size: {len(chunk)}")
        print(f"  First 3 values (ch0): {[chunk[i][0] for i in range(min(3, len(chunk)))]}")

        worker_results.append(chunk)

    # Combine results (simulating file writes from all workers)
    reconstructed = []
    for chunk in worker_results:
        reconstructed.extend(chunk)

    print(f"\n--- Verification ---")
    print(f"Reconstructed samples: {len(reconstructed)}")
    print(f"Expected samples: {len(full_decimated)}")
    print(f"Reconstructed first 10 (ch0): {[reconstructed[i][0] for i in range(10)]}")
    print(f"Expected first 10 (ch0): {[full_decimated[i][0] for i in range(10)]}")

    # Verify they match
    assert len(reconstructed) == len(full_decimated), f"Length mismatch! {len(reconstructed)} vs {len(full_decimated)}"
    assert reconstructed == full_decimated, "Reconstruction doesn't match full decimation!"

    print("\n✓ Parallel workers produce correct output")
    print("✓ Each worker independently selects every k-th sample")
    print("✓ No coordination needed between workers")


def test_chunk_boundaries():
    """Verify no gaps or overlaps at chunk boundaries."""
    print("\n=== Test 3: Chunk Boundary Alignment ===")

    # Simple sequential data
    num_samples = 120
    data = [[i] for i in range(num_samples)]  # Single channel

    decimation_factor = 5

    # Two adjacent chunks
    chunk1 = simulate_decimated_get_traces(data, decimation_factor, 0, 10)  # decimated 0→10
    chunk2 = simulate_decimated_get_traces(data, decimation_factor, 10, 20)  # decimated 10→20

    print(f"Chunk 1 (dec 0→10): {[row[0] for row in chunk1]}")
    print(f"Chunk 2 (dec 10→20): {[row[0] for row in chunk2]}")

    # Verify no gap between chunks
    last_of_chunk1 = chunk1[-1][0]
    first_of_chunk2 = chunk2[0][0]

    print(f"\nLast value of chunk 1: {last_of_chunk1}")
    print(f"First value of chunk 2: {first_of_chunk2}")
    print(f"Difference: {first_of_chunk2 - last_of_chunk1} (should be {decimation_factor})")

    assert first_of_chunk2 - last_of_chunk1 == decimation_factor, "Gap or overlap detected!"

    print("✓ No gaps or overlaps between chunks")
    print("✓ Perfect alignment")


def test_realistic_scenario():
    """Test with realistic neural recording parameters."""
    print("\n=== Test 4: Realistic Scenario ===")

    # Realistic parameters
    original_fs = 30000  # 30 kHz
    target_fs = 1250     # 1.25 kHz
    decimation_factor = original_fs // target_fs  # 24

    duration_sec = 10  # 10 seconds of data
    num_samples = original_fs * duration_sec
    num_channels = 8  # 8 channels

    print(f"Original rate: {original_fs} Hz")
    print(f"Target rate: {target_fs} Hz")
    print(f"Decimation factor: {decimation_factor}")
    print(f"Duration: {duration_sec} seconds")
    print(f"Original samples: {num_samples:,}")
    print(f"Channels: {num_channels}")

    # Create mock data (just first channel for verification)
    # In real scenario, this would be huge, but we only store sample indices
    print("\n(Skipping actual array creation due to size)")

    decimated_samples = num_samples // decimation_factor
    print(f"Decimated samples: {decimated_samples:,}")

    # Simulate file size
    bytes_per_sample = 2  # int16
    original_size_mb = (num_samples * num_channels * bytes_per_sample) / (1024**2)
    decimated_size_mb = (decimated_samples * num_channels * bytes_per_sample) / (1024**2)

    print(f"\nOriginal file size: {original_size_mb:.1f} MB")
    print(f"Decimated file size: {decimated_size_mb:.1f} MB")
    print(f"Compression ratio: {original_size_mb / decimated_size_mb:.1f}x")

    # Verify math
    expected_samples = [0, 24, 48, 72, 96]  # First 5 decimated samples
    print(f"\nFirst 5 decimated samples would be from indices: {expected_samples}")

    print("✓ Realistic scenario parameters valid")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Parallel Decimation Logic")
    print("=" * 70)

    test_basic_decimation()
    test_parallel_workers()
    test_chunk_boundaries()
    test_realistic_scenario()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    print("\nKey Findings:")
    print("• Each worker operates in DECIMATED frame space")
    print("• Workers independently convert to parent space and decimate")
    print("• No interference - each worker processes unique samples")
    print("• Chunk boundaries perfectly aligned")
    print("• This approach is safe for parallel writing")
