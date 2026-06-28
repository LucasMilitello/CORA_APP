# CORA - Cell Open-Region Analyzer

CORA is a desktop application for wound-healing image analysis. It helps load microscopy image groups, segment open regions, review or edit masks, and export quantitative results.

The project includes a Tkinter interface used by the packaged Windows executable and an experimental PySide6/Qt entry point.

## Main Features

- Automatic image loading and grouping by time point.
- Wound/open-region segmentation pipeline inspired by MATLAB-style image processing.
- ROI and mask review tools.
- Batch processing for multiple image groups.
- Export of measurements and processed outputs.
- Optional PyInstaller build script for generating a Windows executable.

## Project Structure

```text
cora_projeto/              Main application package
cora_projeto/services/     Processing, grouping, and export services
cora_projeto/pages/        Tkinter UI pages
cora_projeto/qt/           Experimental PySide6/Qt interface
tools/                     Utility scripts
CORA_testes/               Optional robotized and performance test workspace
requirements.txt           Python dependencies
CORA.spec                  PyInstaller build configuration
build_exe.ps1              Windows build script
launch-qt.ps1              Helper script for the Qt interface
```

## Requirements

- Python 3.12 or newer recommended
- Windows recommended for the desktop executable build
- PowerShell for the included helper scripts

Install the Python dependencies from `requirements.txt`.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Application

Run the main Tkinter application:

```powershell
python -m cora_projeto.run_cora_gui
```

Run the experimental Qt interface:

```powershell
python -m cora_projeto.qt.init_qt
```

Or use the helper script:

```powershell
.\launch-qt.ps1
```

## Building the Windows Executable

The executable build uses PyInstaller and the included `CORA.spec` file.

From the project root, run:

```powershell
.\build_exe.ps1 -Clean
```

The generated executable is written to:

```text
dist/CORA.exe
```

The build process uses the following icon assets:

```text
Logo_GIM_02.png
Logo_GIM_02_icon.png
Logo_GIM_02.ico
```

## Optional Tests

The `CORA_testes/` folder contains robotized and performance test utilities. These tests are intended for development and may require Windows, `pywinauto`, and local test image folders.

Install the extra robotized test dependencies from:

```powershell
python -m pip install -r CORA_testes\requirements_robotizado.txt
```

Some test scripts currently reference local paths and may need adjustment before running on another machine.

## Files Not Committed to Git

The following files and folders are generated locally and should not be committed:

```text
.venv/
__pycache__/
build/
dist/
.cora_tmp/
_cora_resultados/
*.zip
*.log
```

They are ignored because they are environment-specific, temporary, generated during execution, or too large for source control.

## Reproducing the Project After Cloning

After cloning the repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m cora_projeto.run_cora_gui
```

## License

No license file is currently included. Add a `LICENSE` file before publishing if you want to define how others may use, modify, or redistribute this project.
