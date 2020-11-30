## Predicting Critical Transitions in Multiscale Dynamical Systems Using Reservoir Computing



### Get started
-----------

Here is an example to run the algorithm on the time series of Example 1:

```python3 driver.py --name Example1 --n_ens 50 --trainlen 3700 --valid 10 --future 400 --trainbeg 0 --num_layer 1 --n_res_gridsize 8 --n_res_start 600 --n_res_gap 20 --spec_rad_start 0.7 --spec_rad_gap 0.1 --sparsity_start 0.1 --sparsity_gap 0.1 --noise_start 0.001 --noise_gap 0.001```

Here is an example to run the algorithm on the time series of Example 2:

```python3 driver.py --name Example2 --n_ens 50 --trainlen 9410 --valid 4 --future 600 --trainbeg 0 --num_layer 3 --n_res_gridsize 5 --n_res_start 100 --n_res_gap 50 --n_res_start_2 100 --n_res_gap_2 50 --n_res_start_3 100 --n_res_gap_3 50 --spec_rad_start 0.6 --spec_rad_gap 0.05 --spec_rad_start_2 0.7 --spec_rad_gap_2 0.05 --spec_rad_start_3 0.8 --spec_rad_gap_3 0.05 --sparsity_start 0.05 --sparsity_gap 0.05 --sparsity_start_2 0.05 --sparsity_gap_2 0.05 --sparsity_start_3 0.05 --sparsity_gap_3 0.05 --noise_start 0.003 --noise_gap 0.003```

Here is an example to run the algorithm on the time series of Example 3:

```python3 driver.py --name Example3 --n_ens 50 --trainlen 8233 --valid 5 --future 40 --trainbeg 200 --num_layer 1 --n_res_gridsize 7 --n_res_start 500 --n_res_gap 50 --spec_rad_start 0.95 --spec_rad_gap 0.05 --sparsity_start 0.1 --sparsity_gap 0.05 --noise_start 0.003 --noise_gap 0.001```



### Reference
----------
ArXiv: [https://arxiv.org/abs/1908.03771](https://arxiv.org/abs/1908.03771)

Journal: Chaos: An Interdisciplinary Journal of Nonlinear Science (2020); doi: 10.1063/5.0023764