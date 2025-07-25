# README

## Requirements

This project requires **Python 3.9**. To set up the environment and install dependencies, execute:

```bash
pip install -r requirements.txt
```

## Running Experiments
All experiments are executed via the `main.py` script. 

### Running the script
To start the experiments, run the following command:

```bash
python main.py
```

### Results

The results of each experiment are automatically saved in the `results/` folder. Before running an experiment, the script checks if the corresponding results already exist in the `results/` folder. If results are found, the experiment is skipped to avoid redundant computations.


If you want to rerun all experiments and generate new results from scratch, delete the `results/` folder.

### Plots

All the code to generate the report's plots can be found in `plots.ipynb`