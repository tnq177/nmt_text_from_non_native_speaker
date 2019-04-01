# Transformer in Pytorch

Source code for the Transformer model used in the paper "Neural Machine Translation of Text from Non-Native Speakers" [https://arxiv.org/abs/1808.06267](https://arxiv.org/abs/1808.06267).  Require Python3.6 and Pytorch 1.0.

Pretty much just the transformer model, but the validator validates on different dev sets at a time and keeps best checkpoint for each dev set.

# How to train a new model
Write a config function in ``configurations.py``. Then run ``python3 -m nmt --proto config_name``.  

The best checkpoints are saved in ``nmt/saved_models/model_name``. To decode with a checkpoint: ``python3 -m nmt --proto config_name --mode translate --model-file nmt/saved_models/model_name/checkpoint_name.path --input-file path_to_input_file``.

# References
A lot of code / scripts are borrowed from:  

* http://nlp.seas.harvard.edu/2018/04/03/attention.html
* https://github.com/pytorch/fairseq
* https://github.com/EdinburghNLP/nematus
* https://github.com/moses-smt/mosesdecoder

...
