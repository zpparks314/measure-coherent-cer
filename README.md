# measure-coherent-cer
This will be the main repository for the Spring 2025 thrust of Zachary Parks' thesis research

# XCER-SC Project - Parks, Carignan-Dugas, Dreher

## Directory Structure

### `notebooks/` <span style="color: red;"> THESE NOTEBOOKS ARE TECHNICALLY NOT UPDATED</span>
This directory contains templates of utility Python files and notebooks. It serves as the "ground zero" for the project. The notebooks in this directory start with "0", signifying that it's preferred not to run them in the `notebooks/` directory. Instead, run the notebooks in the `run/` directory, which are (mostly) identical to these.

### `run/`
This directory is used for executing, re-executing, and analyzing XCER project circuits on devices and simulators.

### `results/`
This directory is used for storing significant results and interesting plots generated during the course of the project.

## Getting Started

In `notebooks/`, the file `xcer_funcs.py` serves as the main utility script for generating CER circuits and analysis data. Much of the "behind-the-scenes" happens there. 

Also in `notebooks/` is `datalogger.py`, witch was my quick way of adding dictionariess to a `.txt` file and saving it for record keeping purposes.

Notebooks to execute XCER experiments on devices and simulators are found in `run/` and are named `execute_xcer_device.ipynb` and `execute_xcer_simulation.ipynb`. As they are written, they will generate a random "id" for a set of experiments and store them in a directory (if using a device, it will be found in a directory `{backend_name}/{id}`, if using a simulator  in `simulations/{id}`). If running on a new device, be sure to add a directory in named `run/devices/{name_of_backend}`.

As of right now, I saved simulator results in a top-level directory format: `./simulations/yyyymmdd/{id}/`

Notebooks to re-execute XCER experiments on devices and simulators are found in `run/`, and as well `analysis_xcer_device.ipynb` and `analysis_xcer_simulation.ipynb` for analysis.


