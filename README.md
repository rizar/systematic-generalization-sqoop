# Systematic Generalization: What Is Required and Can It Be Learned

The code used for the experiments in [the paper](https://openreview.net/forum?id=HkezXnA9YX).

### Setup

Clone the repo
```
git clone https://github.com/rizar/systematic-generalization-sqoop.git
cd systematic-generalization-sqoop
export NMN=$PWD
```
Setup the environment using `conda` (recommended) and install this as a package in development mode
```
conda env create -f environment.yml
conda activate sysgen
pip install -e .
```
if you don't use conda, you can do `pip install --user -r requirements.txt`


Download all versions of SQOOP dataset from [here](https://drive.google.com/file/d/1yaXQL-MH0nQM9cqRbIrWkB3kBNM_ltY_/view?usp=sharing)
and unpack it. Let `$DATA` be the location of the data on your system.

### Running Experiments

In the examples below we are using SQOOP with `#rhs/lhs=1`, other versions can be used by changing `--data_dir`.

#### FiLM

    scripts/train/film_flatqa.sh --data_dir $DATA/sqoop-variety_1-repeats_30000 --checkpoint_path model.pt\
     --num_iterations 200000

#### MAC

    scripts/train/mac_flatqa.sh --data_dir $DATA/sqoop-variety_1-repeats_30000 --checkpoint_path model.pt\
     --num_iterations 100000

#### Conv+LSTM

    scripts/train/convlstm_flatqa.sh --data_dir $DATA/sqoop-variety_1-repeats_30000 --checkpoint_path model.pt\
     --num_iterations 200000

#### RelNet

    scripts/train/rel_flatqa.sh --data_dir $DATA/sqoop-variety_1-repeats_30000 --checkpoint_path model.pt\
     --num_iterations 500000

#### NMN-Tree, NMN-Chain, NMN-Chain-Shortcut

    scripts/train/shnmn_flatqa.sh --data_dir $DATA/sqoop-variety_1-repeats_30000\
     --hard_code_tau --tau_init tree --hard_code_alpha --alpha_init correct\
     --num_iterations 50000 --checkpoint_path model.pt

For a different layout use `--tau_init=chain` or `--tau_init=chain_shortcut`. For a different module, use `--use_module=find`, the default is Residual.
Make sure to train for 200000 iterations if you use Find.

#### Stochastic-N2NMN

     scripts/train/shnmn_flatqa.sh --data_dir $DATA/sqoop-variety_1-repeats_30000\
      --shnmn_type hard --model_bernoulli 0.5 --hard_code_alpha --alpha_init=correct\
      --num_iterations 200000 --checkpoint_path model.pt

`--model_bernoulli` is the initial probability of the model being a tree.

#### Attention-N2NMN

    scripts/train/shnmn_flatqa.sh --data_dir $DATA/sqoop-variety_1-repeats_30000\
     --hard_code_tau --tau_init tree --use_module=find --num_iterations 200000\
     --checkpoint_path model.pt

### Citation

**Bahdanau, D., Murty, S.**, Noukhovitch, M., Nguyen, T. H., de Vries, H., & Courville, A. (2018). Systematic Generalization: What Is Required and Can It Be Learned?. ICLR 2019

(the first two authors contributed equally)

```
@inproceedings{sysgen2019,
    title = {Systematic Generalization: What Is Required and Can It Be Learned?},
    booktitle = {International Conference on Learning Representations},
    author = {Bahdanau, Dzmitry and Murty, Shikhar and Noukhovitch, Michael and Nguyen, Thien Huu and Vries, Harm de and Courville, Aaron},
    year = {2019},
    url = {https://openreview.net/forum?id=HkezXnA9YX},
}
```

### Acknowledgements.

This code is based on the reference implementation for ["FiLM: Visual Reasoning with a General Conditioning Layer"](https://github.com/ethanjperez/film) by Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville (AAAI 2018) which was based on the reference implementation for ["Inferring and Executing Programs for Visual Reasoning"](https://github.com/facebookresearch/clevr-iep) by Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Judy Hoffman, Fei-Fei Li, Larry Zitnick, Ross Girshick (ICCV 2017)
