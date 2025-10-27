
# cod-accra: Photochem.clima

This folder reproduces all `Photochem.clima` calculations for the cod-accra model inter-comparison. To run this code, first use the `conda` package manager to create a new environment:

```sh
conda create -n codaccra -c conda-forge photochem=0.6.7 matplotlib
conda activate codaccra
```

Next, run the main function:

```sh
python main.py
```

After a ~1 minute of runtime, the results will appear in the `results/` folder as .txt files.
