# Video Analysis Final Project

This project extracts GCP video features, computes rule-based scene interpretation, adds LVLM semantic interpretation, and supports single-video and batch execution through `gcp`, `lvlm`, and `full` modes.

## Installation guide

The project requires Python 3.10 or newer and is tested with Python 3.12.

The [`requirements.txt`](requirements.txt) file contains only the installable Python dependency entries. All installation and setup instructions are documented below.

### 1. Install FFmpeg

FFmpeg is required for video probing and frame extraction. The `ffmpeg-python` package does not install the FFmpeg system executable.

Windows PowerShell:

```powershell
winget install Gyan.FFmpeg
```

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

Restart the terminal if necessary, then verify the installation:

```bash
ffmpeg -version
```

### 2. Create and activate a virtual environment

Run the commands from the repository root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file installs:

- The OpenAI-compatible client used with the Z.ai GLM API.
- FFmpeg Python bindings.
- Google Cloud Storage and Video Intelligence clients.
- Google API and HTTP dependencies used by retry handling.

### 4. Configure Google Cloud

For `gcp` and `full` modes:

1. Create or select a Google Cloud project.
2. Enable Google Cloud Storage and Video Intelligence.
3. Create a Cloud Storage bucket.
4. Create a service account with access to the bucket and Video Intelligence.
5. Download its JSON credential file and keep it outside Git.

### 5. Create the environment file

Create `.env` in the repository root:

```dotenv
GLM_API_KEY=your_z_ai_api_key
DEFAULT_BUCKET_NAME=your_gcs_bucket_name
DEFAULT_GCS_PREFIX=uploads
GOOGLE_APPLICATION_CREDENTIALS=C:/absolute/path/to/service-account.json
```

`ZHIPU_API_KEY` can be used instead of `GLM_API_KEY`. Both `.env` and the local service-account credential file must remain private and must not be committed.

### 6. Verify the installation

Run the offline test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Verify that the project compiles:

```bash
python -m compileall -q src tests main.py
```

## Running the project

Complete single-video pipeline:

```bash
python -m src.pipeline.run_full_pipeline --video "videos/videos/ACCEDE09230.mp4" --mode full
```

GCP features and algorithm only:

```bash
python -m src.pipeline.run_full_pipeline --video "videos/videos/ACCEDE09230.mp4" --mode gcp
```

LVLM using an existing `VideoFeatures.json`:

```bash
python -m src.pipeline.run_full_pipeline --video "videos/videos/ACCEDE09230.mp4" --mode lvlm
```

Batch execution:

```bash
python -m src.pipeline.run_batch_pipeline --videos_dir "videos/videos" --mode full
```

Update the LVLM interaction evaluation table:

```bash
python -m src.pipeline.collect_lvlm_interaction
```

Generated files are stored under `outputs/<video_id>/` as `VideoFeatures.json` and `VideoInterpretation.json`.
