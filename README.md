# Grammattical-Graph-Neural-Network
# Wesfeiler-and-Lehman-go-grammatical


This code is related to the submited paper Wesfeiler-and-Lehman-go-grammatical in Neurips 2023. It uses pytorch.


To reproduce the experiments in the paper, you must launch the file using python.


QM9 experiment:

In qm9_dataset.py, ntask corresponds to the target you want to learn from 0 to 11.

After choosing a target, launch qm9_dataset.py.

TUD experiment:

In TU_dataset.py, change the configuration of G2N2 according to the config in the supplementary material of the paper.
Name correspond to the dataset you which to use.

After that launch TU_dataset.py.
To see the results, launch TUD_result.py

