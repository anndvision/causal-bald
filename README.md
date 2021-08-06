# causal-bald

## Installation

```.sh
$ git clone git@github.com:[anon]/causal-bald.git
$ cd causal-bald
$ conda env create -f environment.yml
$ conda activate causal-bald
```
[Optional] For developer mode
```.sh
$ pip install -e .
```

## Example: Active learning IHDP

### Active learning loop

```.sh
causal-bald \
    active-learning \
        --job-dir experiments/ \
        --num-trials 5 \
        --step-size 10 \
        --warm-start-size 100 \
        --max-acquisitions 38 \
        --acquisition-function mu-rho \
        --temperature 0.25 \
        --gpu-per-trial 0.2 \
    ihdp \
        --root assets/ \
    deep-kernel-gp
```

### Evaluation

Evaluate PEHE at each acquisition step

```.sh
causal-bald \
    evaluate \
        --experiment-dir experiments/active_learning/ss-10_ws-100_ma-38_af-random_temp-1.0/ihdp/deep_kernel_gp/kernel-Matern32_ip-100-dh-200_do-1_dp-3_ns--1.0_dr-0.1_sn-0.95_lr-0.001_bs-100_ep-1000/ \
        --output-dir experiments/ihdp/ \
    pehe
```

Plot results

```.sh
causal-bald \
    evaluate \
        --experiment-dir experiments/ihdp/ \
    plot-convergence \
        -m mu-rho

```
