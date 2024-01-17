This code is related to the Wesfeiler-and-Lehman-go-grammatical paper submitted to ICLR 2024. It uses pytorch and cuda.

Python version 3.8.10
Pytorch version 1.12.0


To reproduce the experiments in the paper, you will need to run the file using Python.


QM9 experiment:

In qm9_dataset.py, ntask is the target you want to learn from 0 to 11.

After choosing a target, run qm9_dataset.py.

QM9 12 targets:

run qm9_dataset_12_labels.py

To use the GNN derived from the exhaustive CFG run qm9_dataset_12_labels_exhaust_GNN.py

TUD experiment:

In TU_dataset.py change the configuration of G2N2 according to the config in the supplementary material of the paper.
The "Name" variable corresponds to the dataset you want to use.

Then run TU_dataset.py.
To see the results, run TUD_results.py.

Filtering experiment:

In G2N2_filtering.py, select the type of filter to learn with ntask( 0 : low-pass, 1 : high-pass, 2 : band-pass).

Then run G2N2_filtering.py.

