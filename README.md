# FiLM: Visual Reasoning with a General Conditioning Layer

## Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville

This code implements a Feature-wise Linear Modulation approach to Visual Reasoning - answering multi-step questions on images. This codebase reproduces results from the AAAI 2018 paper "FiLM: Visual Reasoning with a General Conditioning Layer," which extends prior work "Learning Visual Reasoning Without Strong Priors" presented at ICML's MLSLP workshop.

### Code Outline

This code is a fork from the code for "Inferring and Executing Programs for Visual Reasoning" available [here](https://github.com/facebookresearch/clevr-iep).

Our FiLM Generator is located in [vr/models/film_gen.py](https://github.com/ethanjperez/sa-iep/blob/master/vr/models/film_gen.py), and our FiLMed Network and FiLM layer implementation is located in [vr/models/filmed_net.py](https://github.com/ethanjperez/sa-iep/blob/master/vr/models/filmed_net.py).

We inserted a new model mode "FiLM" which integrates into forked code for [CLEVR baselines](https://arxiv.org/abs/1612.06890) and the [Program Generator + Execution Engine model](https://arxiv.org/abs/1705.03633). Throughout the code, for our model, our FiLM Generator acts in place of the "program generator" which generates the FiLM parameters for an the FiLMed Network, i.e. "execution engine." In some sense, FiLM parameters can vaguely be thought of as a "soft program" of sorts, but we use this denotation in the code to integrate better with the forked models.

### Setup and Training

Because of this integration, setup instructions for the FiLM model are nearly the same as for "Inferring and Executing Programs for Visual Reasoning." We will post more detailed instructions on how to use our code in particular soon for more step-by-step guidance. For now, the guidelines below should give substantial direction to those interested.

First, follow the virtual environment setup [instructions](https://github.com/facebookresearch/clevr-iep#setup).

Second, follow the CLEVR data preprocessing [instructions](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#preprocessing-clevr).

Lastly, model training details are similar at a high level (though adapted for FiLM and our repo) to [these](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#training-on-clevr) for the Program Generator + Execution Engine model, though our model only uses one step of training, rather than a 3-step training procedure.

The below script has the hyperparameters and settings to reproduce FiLM CLEVR results:
```bash
sh scripts/train/film.sh
```


For CLEVR-Humans, data preprocessing instructions are [here](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#preprocessing-clevr-humans).
The below script has the hyperparameters and settings to reproduce FiLM CLEVR-Humans results:
```bash
sh scripts/train/film_humans.sh
```


Training a CLEVR-CoGenT model is very similar to training a normal CLEVR model. Training a model from pixels requires modifying the preprocessing with scripts included in the repo to preprocess pixels. The scripts to reproduce our results are also located in the scripts/train/ folder.

We tried to not break existing models from the CLEVR codebase with our modifications, but we haven't tested their code after our changes. We recommend using using the CLEVR and "Inferring and Executing Programs for Visual Reasoning" code directly.

Training a solid FiLM CLEVR model should only take ~12 hours on a good GPU (See training curves in the paper appendix).

### Running models

We added an interactive command line tool for use with the below command/script. It's actually super enjoyable to play around with trained models. It's great for gaining intuition around what various trained models have or have not learned and how they tackle reasoning questions.
```bash
python run_model.py --program_generator <FiLM Generator filepath> --execution_engine <FiLMed Network filepath>
```

By default, the command runs on [this CLEVR image](https://github.com/ethanjperez/sa-iep/blob/master/img/CLEVR_val_000017.png) in our repo, but you may modify which image to use via command line flag to test on any CLEVR image.

CLEVR vocab is enforced by default, but for CLEVR-Humans models, for example, you may append the command line flag option '--enforce_clevr_vocab 0' to ask any string of characters you please.

In addition, one easier way to try out zero-shot with FiLM is to run a trained model with run_model.py, but with the implemented debug command line flag on so you can manipulate the FiLM parameters modulating the FiLMed network during the forward computation. For example, '--debug_every -1' will stop the program after the model generates FiLM parameters but before the FiLMed network carries out its forward pass using FiLM layers.

Thanks for stopping by, and we hope you enjoy playing around with FiLM!

### Bibtex
```bash
@InProceedings{perez2018film,
  title={FiLM: Visual Reasoning with a General Conditioning Layer},
  author={Ethan Perez and Florian Strub and Harm de Vries and Vincent Dumoulin and Aaron C. Courville},
  booktitle={AAAI},
  year={2018}
}
```
