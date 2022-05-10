# Weak Labeler


The aim of the project is to classify the class of a given text within the designaed provided classes. 

Before proceeding further please install the module for further use:

```
python setup.py install
```

or a better alternative

```
pip install -U .
```

The codebase is split into several submodules

1) tools - utilities split for each particular use-case
2) zeroShot&prompts - This folder contains the experiments with zero-shot classification and prompts.
3) few-shot - This folder contains the experiments with
4) tools/sentence_scorer.py - the module for assessing the naturality of the sentence

All the solvers contain a standard script for loading the data, training and evaluating/predicting.

An example use of the few-shot pipeline can look like this

```
python weaklabeler/fewShot/train.py --experiment_name fewShot_currated --feat_extractor roberta-base --training_path weaklabeler/Data/covid_currated.csv --model_save_path weaklabeler/models/ --batch_size 16 --epochs 3 --learning_rate 0.1 --shot_num 250 --val_step 100 --text_col TEXT --target_col Target --target_config_path weaklabeler/configs/covid_currated_config.json 

```

In order to use the the zero-shot or prompt based classifers several `configs` have to be opened, please consult the example under the `weaklabeler/configs`. An example for annotation looks like
the follwoing:

```
python main.py --model_name vicgalle/xlm-roberta-large-xnli-anli --dataset_path Data/example_~100_variation.json --config_path configs/populated_zero.json  --labeler_type zero

```



## Codebase

The codebase is fully documented and optimised for parallel processing across various nodes and gpu's. The Deep Learning modules are completely differentiable with a complete experimental architecture and hyperparameter tracking integrated within the codebase.

## Results

TODO

___
## Experiment Tracking

For granular experiment tracking and filtering, I use `Aim`. This allows for filtering between various experimental settings, architectures etc.