# Predicting Critical Transitions in Multiscale Dynamical Systems Using Reservoir Computing



## Get started
-----------
### Description:

```option``` = option of running the baseline (0) or our algorithm (1) (see the paper in Reference below)

```n_ens``` = number of networks independently trained for ensemble learning (denoted <img src="https://render.githubusercontent.com/render/math?math=N_{ens}"> in the paper)

```trainlen``` + 1 = length of accessible/given time series (denoted <img src="https://render.githubusercontent.com/render/math?math=N">+1 in the paper)

```valid``` = length of validation time series (denoted <img src="https://render.githubusercontent.com/render/math?math=N_{\nu}"> in the paper)

```future``` = length of predicted time series (denoted <img src="https://render.githubusercontent.com/render/math?math=M"> in the paper)

```trainbeg``` = index indicating the initial time (denoted <img src="https://render.githubusercontent.com/render/math?math=t_0"> in the paper) on which the given time series runs

```num_layer``` = number of layer in the network (denoted <img src="https://render.githubusercontent.com/render/math?math=L">+1 in the paper)

```dt``` = time step used in numerical integration generating the time series data (denoted <img src="https://render.githubusercontent.com/render/math?math=\Delta t"> in the paper)

so that length of time series used for training in the case of ```option``` = 0 is ```trainlen``` - ```valid``` + 1 and that in the case of ```option``` = 1 is ```trainlen``` - ```valid```


### Hyperparameter grid search: 

For hyperparameters in the first layer of network:

- dimension of reservoir hidden states is selected from the interval [```n_res_start_1```, ```n_res_start_1``` + ```n_res_total``` * ```n_res_delta_1```)

- spectral radius is selected from the interval [```spec_rad_start_1```, ```spec_rad_start_1``` + ```spec_rad_total``` * ```spec_rad_delta_1```)

- sparsity level is selected from the interval [```sparsity_start_1```, ```sparsity_start_1``` + ```sparsity_total``` * ```sparsity_delta_1```)

- noise level is selected from the interval [```noise_start```, ```noise_start``` + ```noise_total``` * ```noise_delta```)

and similarly for the hyperparameters in the second layer and so on

If needed, discard an initial transient by disregarding the first ```n_transient``` reservoir hidden states


### Examples:

- Here is an example to run the algorithm on the time series of Example 1:

```python3 driver.py --name Example1 --option 1 --n_ens 50 --trainlen 3700 --valid 10 --future 400 --trainbeg 0 --num_layer 1 --n_res_start_1 600 --n_res_delta_1 20 --n_res_total 8 --spec_rad_start_1 0.7 --spec_rad_delta_1 0.1 --spec_rad_total 1 --sparsity_start_1 0.1 --sparsity_delta_1 0.1 --sparsity_total 1 --noise_start 0.001 --noise_delta 0.001 --noise_total 1```

- Here is an example to run the algorithm on the time series of Example 2:

```python3 driver.py --name Example2 --option 1 --n_ens 50 --trainlen 9410 --valid 4 --future 590 --trainbeg 0 --num_layer 3 --n_res_start_1 100 --n_res_delta_1 50 --n_res_start_2 100 --n_res_delta_2 50 --n_res_start_3 100 --n_res_delta_3 50 --n_res_total 5 --spec_rad_start_1 0.6 --spec_rad_delta_1 0.05 --spec_rad_start_2 0.7 --spec_rad_delta_2 0.05 --spec_rad_start_3 0.8 --spec_rad_delta_3 0.05 --spec_rad_total 1 --sparsity_start_1 0.05 --sparsity_delta_1 0.05 --sparsity_start_2 0.05 --sparsity_delta_2 0.05 --sparsity_start_3 0.05 --sparsity_delta_3 0.05 --sparsity_total 1 --noise_start 0.003 --noise_delta 0.003 --noise_total 1```

- Here is an example to run the algorithm on the time series of Example 3:

```python3 driver.py --name Example3 --option 1 --n_ens 50 --trainlen 8233 --valid 5 --future 37 --trainbeg 200 --num_layer 1 --n_res_start_1 500 --n_res_delta_1 50 --n_res_total 7 --spec_rad_start_1 0.95 --spec_rad_delta_1 0.05 --spec_rad_total 1 --sparsity_start_1 0.1 --sparsity_delta_1 0.05 --sparsity_total 1 --noise_start 0.003 --noise_delta 0.001 --noise_total 1 --n_transient 100```



## Reference
----------
For more details, please refer to the paper:

- ArXiv: [https://arxiv.org/abs/1908.03771](https://arxiv.org/abs/1908.03771)

- Journal: Chaos: An Interdisciplinary Journal of Nonlinear Science (2020); doi: 10.1063/5.0023764
