# Running the Pipeline on Quest

Useful documentation:

- [Quest file systems and storage](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/filesystem/filesystem.html)
- [Logging in to Quest](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/login/login-quest.html)
- [Slurm job scheduler](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/slurm/slurm.html)
- [Globus collections](https://rcdsdocs.it.northwestern.edu/systems/globus/globus-collection.html)

## 1. Set Up Globus

Use [Globus](https://rcdsdocs.it.northwestern.edu/systems/globus/globus-collection.html) to transfer data between Quest and FSMResFiles or a personal computer.

Transfer recordings data to your scratch space at `/scratch/<netid>`.

## 2. Log In

This guide covers two access methods: a terminal session and a Quest OnDemand desktop session.

### A. Terminal

Open a terminal application. On macOS or Linux, use your usual terminal; on Windows, use PowerShell. Connect to Quest with:

```bash
ssh <netid>@login.quest.northwestern.edu
```

If you use VS Code, there's a Remote - SSH extension, select:

```text
Open a Remote Window → Connect to Host... → Add New SSH Host...
```

Add an entry like this to your SSH configuration:

```text
Host quest
    HostName: login.quest.northwestern.edu
    User: <netid>
    Port 22
```

Enter your Northwestern NetID password when prompted.

For more information, see [Logging in to Quest](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/login/login-quest.html).

### B. Desktop

For a more user-friendly interface, you can connect to a remote desktop through Quest OnDemand.

Open the [Quest OnDemand portal](https://ondemand.quest.northwestern.edu) and sign in with your Northwestern credentials. If you are off campus, connect to the Northwestern VPN (GlobalProtect VPN) first.

In the web portal, select: 
```
My Interactive Sessions -> Quest GNOME Desktop
```

There will be a Slurm resource request form which can be completed using `run_hpc.sh` as a reference. 

A single A100 GPU is usually more than enough for running Kilosort. Using higher end GPU likely won't speed up the progress because Kilosort4 uses 60,000 batch size by default.

Depending on requested resources, the session may remain queued for a while. Once it starts, you can use the GNOME desktop like a Linux workstation.

For more information, see [Quest OnDemand](https://rcdsdocs.it.northwestern.edu/systems/quest/ondemand/ondemand.html).

## 3. Run the Pipeline on Quest

### A. Terminal

Load Git and uv:

```bash
module load git/2.37.2
module load uv/0.8.18
```

Clone the repository and create its environment:

```bash
git clone https://github.com/8Nero/pipeline.git
cd pipeline
uv sync
```

Once your pipeline configuration file is ready, replace the `<account-name>` and `<email>` placeholders, then set `CONFIG_PATH` in `run_hpc.sh`.

Adjust the requested wall time, memory, CPU, and GPU resources if needed.

Submit the Slurm job from the repo directory as:

```bash
sbatch run_hpc.sh
```

Slurm will return a job ID that you can use to check the job's status. You can check the status of your job with:

```bash
squeue -u <netid>
squeue -j <job-id>
```

After the job has completed successfully, check its resource usage with:

```bash
seff <job-id>
```

For more information, see the [Slurm job scheduler guide](https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/slurm/slurm.html).

The Slurm standard output and standard error files are saved as `pipe_<job-id>.out` and `pipe_<job-id>.err` in the directory from which you run `sbatch`.

### B. Desktop

Once your desktop session starts, run the pipeline as usual.

Load the Git and uv versions identified above:

```bash
module load git/2.37.2
module load uv/0.8.18
```

Clone the repository and create its environment:

```bash
git clone https://github.com/8Nero/pipeline.git
cd pipeline
uv sync
```

Run the pipeline:

```bash
uv run pipe /path/to/config.yaml
```
