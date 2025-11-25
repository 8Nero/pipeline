import spikeinterface.core as si
from pathlib import Path

if __name__ == "__main__":
    bin_path = r"E:\pipeline_output\mouse3_hunt-sleepx2\ProbeA\concat"
    rec = si.load(bin_path)

    si.set_global_job_kwargs(n_jobs=16, chunk_duration='2s', mp_context='spawn')

    rec = rec.frame_slice(start_frame=0, end_frame=8 * 60 * 60 * 30000)
    save_path = Path(r"E:\pipeline_output\mouse3_hunt-sleepx2\ProbeA\concat_8h")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rec.save(format='binary',
            folder=save_path,
            overwrite=True)