# Topic-Guided Sampling For Data-Efficient Multi-Domain Stance Detection

This is the official repository for the ACL 2023 "[Topic-Guided Sampling For Data-Efficient Multi-Domain Stance Detection](https://aclanthology.org/2023.acl-long.752.pdf)" paper.

![Tested](https://github.com/copenlu/TESTED/assets/8036160/c3a141ee-c6e2-4534-b626-9424ab501eb9)


The task of Stance Detection is concerned with identifying the attitudes expressed by an author towards a target of interest. This task spans a variety of domains ranging from social media opinion identification to detecting the stance for a legal claim. However, the framing of the task varies within these domains in terms of the data collection protocol, the label dictionary and the number of available annotations. Furthermore, these stance annotations are significantly imbalanced on a per-topic and inter-topic basis. These make multi-domain stance detection challenging, requiring standardization and domain adaptation. To overcome this challenge, we propose Topic Efficient StancE Detection (TESTED), consisting of a topic-guided diversity sampling technique used for creating a multi-domain data efficient training set and a contrastive objective that is used for fine-tuning a stance classifier using the produced set. We evaluate the method on an existing benchmark of 16 datasets with in-domain, i.e. all topics seen and out-of-domain, i.e. unseen topics, experiments. The results show that the method outperforms the state-of-the-art with an average of 3.5 F1 points increase in-domain and is more generalizable with an averaged 10.2 F1 on out-of-domain evaluation while using <10% of the training data. We show that our sampling technique mitigates both inter- and per-topic class imbalances. Finally, our analysis demonstrates that the contrastive learning objective allows the model for a more pronounced segmentation of samples with varying labels.

***Update***: We managed to currently host the datasets proposed and processed by [Hardalov et al.](https://arxiv.org/abs/2104.07467) seperately [here]
(https://drive.google.com/drive/folders/1jxqgiCGwzNkkRa9THN7vZH-nzJwjHVx1?usp=drive_link). The link also contains `stance_data.csv` along with the testing files for used in the evaluation.

# Citations
```
@inproceedings{arakelyan-etal-2023-topic,
    title = "Topic-Guided Sampling For Data-Efficient Multi-Domain Stance Detection",
    author = "Arakelyan, Erik  and
      Arora, Arnav  and
      Augenstein, Isabelle",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.752",
    doi = "10.18653/v1/2023.acl-long.752",
    pages = "13448--13464",
    abstract = "The task of Stance Detection is concerned with identifying the attitudes expressed by an author towards a target of interest. This task spans a variety of domains ranging from social media opinion identification to detecting the stance for a legal claim. However, the framing of the task varies within these domains in terms of the data collection protocol, the label dictionary and the number of available annotations. Furthermore, these stance annotations are significantly imbalanced on a per-topic and inter-topic basis. These make multi-domain stance detection challenging, requiring standardization and domain adaptation. To overcome this challenge, we propose Topic Efficient StancE Detection (TESTED), consisting of a topic-guided diversity sampling technique used for creating a multi-domain data efficient training set and a contrastive objective that is used for fine-tuning a stance classifier using the produced set. We evaluate the method on an existing benchmark of 16 datasets with in-domain, i.e. all topics seen and out-of-domain, i.e. unseen topics, experiments. The results show that the method outperforms the state-of-the-art with an average of 3.5 F1 points increase in-domain and is more generalizable with an averaged 10.2 F1 on out-of-domain evaluation while using {\textless}10{\%} of the training data. We show that our sampling technique mitigates both inter- and per-topic class imbalances. Finally, our analysis demonstrates that the contrastive learning objective allows the model for a more pronounced segmentation of samples with varying labels.",
}
```


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
python weaklabeler/fewShot/train.py --experiment_name fewShot_epochs=5_n=8000_contrastive=True_ood=True --feat_extractor roberta-large --training_path weaklabeler/Data/stance_data/stance_data.csv --model_save_path weaklabeler/models/ --valid_path weaklabeler/Data/stance_data/stance_test_data_sample.csv --num_labels 4 --batch_size 4 --epochs 5 --learning_rate 0.00001 --shot_num 8000 --val_step 1 --text_col text --target_col label_name --target_config_path weaklabeler/configs/few_shot_diverse_sample.json --contrastive --filter_out dataset_name/iac
```


___
## Experiment Tracking

For granular experiment tracking and filtering, we use `Aim`. This allows for filtering between various experimental settings, architectures etc.
