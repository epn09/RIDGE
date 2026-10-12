# RIDGE

[![DOI](https://zenodo.org/badge/1414969339.svg)](https://doi.org/10.5281/zenodo.23327904)

## Tutorial: Detecting levees with the RIDGE framework

### Step 0: Preparation

Compile the Cython code:
```bash
./init_cython.sh
```

### Step 1: Calculate the required parameters with Google Earth Engine (GEE)

```bash
python LeveeIndex.py tutorial
```
Wait for the task to finish in GEE, then download the output file and save it as `output/tutorial_input_vars.tif`.

### Step 2: Trace the lines

```bash
python spanning_tree.py output/tutorial_input_vars.tif output/tutorial.gpkg
```
The output is saved to `output/tutorial.gpkg`. See the Python script for additional options.

### Step 3: Filtering

Use your preferred method to filter levee lines from the traced lines.

To use the Bayesian inference approach described in the paper, download `annotated_lines.gpkg` from <https://doi.org/10.5281/zenodo.19199679> and save it in `model`. Then run:
```bash
python bayesian_inference.py
```

The model parameters are saved to `model/model_params.nc`.

## References
- Khanh, D. N., TBA
