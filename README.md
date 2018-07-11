# On Systematic Compositionality of Visual Question Answering Models

TODO hyperlinks

The paper draft (WIP): https://www.overleaf.com/16233948sbvjhbzwscjn#/62126987/

This code allows to train a number of VQA models on CLEVR, SHAPES and also on a family of custom FlatQA datasets.
The models available include:

- (Homogeneous-) Neural Module Networks (a.k.a. as Execution Engine (EE)) from Johnson et al, 2017
- (Heterogeneous-) Neural Module Networks from Andreas et al, 2016
- Compositional Attention Networks (a.k.a. Memory-Attention-Control (MAC)) from Hudson et al, 2010
- FiLM from Perez et al, 2018
- FiLM with attention, inspired by Strub et al, 2018, but implemented somewhat differently

## Installation

Clone the repository. Create a Conda environment and install the code in the environment in development mode:

```
conda env create -f environment.yaml
source activate nmn
pip install -e . 
```

(if you run this on MILA servers, environment creation will take some time ...)

Any time you want run the code, activate the environment first:

```
source activate nmn
```

When you are done, deactivate the environment:

```
source deactivate
```

## Training Models

Let `$ROOT` be the path to your checkout. A typical training commmands looks as follows:

```
bash $ROOT/nmn-iwp/scripts/train/film_flatqa.sh  --data_dir /data/milatmp1/bahdanau/data/flatqa/relations_BelowSquare
```

Make sure to have a GPU when you run the code, cause it doesn't work just CPU.

##  Datasets

Most FlatQA datasets: `/data/milatmp1/bahdanau/data/flatqa/relations_BelowSquare`
FlatQA-Letters: `/data/milatmp1/noukhovm/cedar/flatqa-letters/`
CLEVR: `/data/milatmp1/bahdanau/data/clevr`

## Generating Datasets

See `scripts/dataset/generate_flatqa.py`, `scripts/dataset/generate_flatqa_letters.py`.

## Visualing Datasets and Models

We inspect datasets and models with IPython notebooks which we don't normally keep under version control
(they are a bit of a pain to deal with). Inspiration can be drawn from:

```
/data/milatmp1/bahdanau/FlatQA_Relations_Results.ipynb
/data/milatmp1/bahdanau/Explore_GridQA.ipynb
```

### Acknowledgements.

This code is based on the reference implementation for "FiLM: Visual Reasoning with a General Conditioning Layer" by Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville.
