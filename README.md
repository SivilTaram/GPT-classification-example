# GPT-classification-finetune

An example for fine-tuning GPT model on classification tasks. Concretely, on glue/cola task. The only dependency is the awesome library [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT), and the code is mainly from their repo, thanks to their efforts!

## How to Start

Our project is tested on python3, so first of all, you should install python3 on your server. We recommend the virtualenv to achieve it. 

Then, you should install the dependencies including `jupyter`, `pytorch` and `pytorch-pretrained-BERT`. We provide a `requirement.txt` file for convenience. You could simply install the dependencies using:

```shell
pip install requirements.txt
```

## Prepare Data

You could simpley use `python download_glue.py` to automatically download data and extract them into `glue` folder.

## Finetune Model

After installing all dependencies, you could either start jupyter notebook or finetune GPT using linux/windows batch file. The files `train.ipynb`, `linux_run_script.sh` and `windows_run_script.bat` are prepared for you.

You could directly run script like shell in linux or batch file in windows under your terminal like:

```shell
./linux_run_script.sh
```

Or you could start jupyter notebook and open the `train.ipynb` to start the finetuning process inside it.

```shell
jupyter notebook
```

## Some Details

We implement the classification on the basis of GPT via the BERT style, which means we add some new special tokens like `_classify_`. **It is different from the original GPT paper**. 

## Experiment Result

On the tested CoLA task, we achieve the same result as reported in Radford et.al 2018. Compared with the number reported in the paper `45.4`, our code could reproduce the mcc as `50.3`. Here are some training logs:

```log
07/09/2019 10:44:47 - INFO - __main__ -   device: cuda n_gpu: 1, distributed training: False, 16-bits training: False
07/09/2019 10:44:50 - INFO - pytorch_pretrained_bert.tokenization_openai -   loading vocabulary file https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-vocab.json from cache at C:\Users\v-qianl\.pytorch_pretrained_bert\4ab93d0cd78ae80e746c27c9cd34e90b470abdabe0590c9ec742df61625ba310.b9628f6fe5519626534b82ce7ec72b22ce0ae79550325f45c604a25c0ad87fd6
07/09/2019 10:44:50 - INFO - pytorch_pretrained_bert.tokenization_openai -   loading merges file https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-merges.txt from cache at C:\Users\v-qianl\.pytorch_pretrained_bert\0f8de0dbd6a2bb6bde7d758f4c120dd6dd20b46f2bf0a47bc899c89f46532fde.20808570f9a3169212a577f819c845330da870aeb14c40f7319819fce10c3b76
07/09/2019 10:44:50 - WARNING - pytorch_pretrained_bert.tokenization_openai -   ftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy.
07/09/2019 10:44:50 - INFO - pytorch_pretrained_bert.tokenization_openai -   Special tokens {'_start_': 40478, '_delimiter_': 40479, '_classify_': 40480}
07/09/2019 10:44:52 - INFO - pytorch_pretrained_bert.modeling_openai -   loading weights file https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-pytorch_model.bin from cache at C:\Users\v-qianl\.pytorch_pretrained_bert\e45ee1afb14c5d77c946e66cb0fa70073a77882097a1a2cefd51fd24b172355e.e7ee3fcd07c695a4c9f31ca735502c090230d988de03202f7af9ebe1c3a4054c
07/09/2019 10:44:52 - INFO - pytorch_pretrained_bert.modeling_openai -   loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-config.json from cache at C:\Users\v-qianl\.pytorch_pretrained_bert\a27bb7c70e9002d7558d2682d5a95f3c0a8b31034616309459e0b51ef07ade09.f59b19eb0e361a0230a1106b66b8c6e7a994cb200cd63d9190cda8d56d75ff85
07/09/2019 10:44:52 - INFO - pytorch_pretrained_bert.modeling_openai -   Model config {
  "afn": "gelu",
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 512,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 512,
  "n_special": 0,
  "resid_pdrop": 0.1,
  "vocab_size": 40478
}

07/09/2019 10:44:59 - INFO - __main__ -   ***** Running training *****
07/09/2019 10:44:59 - INFO - __main__ -     Num examples = 8551
07/09/2019 10:44:59 - INFO - __main__ -     Batch size = 32
07/09/2019 10:44:59 - INFO - __main__ -     Num steps = 804
Iteration: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 268/268 [05:04<00:00,  1.01s/it]
Iteration: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 268/268 [05:19<00:00,  1.04s/it]
Iteration: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 268/268 [05:20<00:00,  1.00it/s]
07/09/2019 11:00:46 - INFO - pytorch_pretrained_bert.modeling_openai -   loading weights file experiment/CoLA\pytorch_model.bin
07/09/2019 11:00:46 - INFO - pytorch_pretrained_bert.modeling_openai -   loading configuration file experiment/CoLA\config.json
07/09/2019 11:00:46 - INFO - pytorch_pretrained_bert.modeling_openai -   Model config {
  "afn": "gelu",
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 512,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 512,
  "n_special": 3,
  "resid_pdrop": 0.1,
  "vocab_size": 40478
}

07/09/2019 11:00:51 - INFO - __main__ -   ***** Running evaluation *****
07/09/2019 11:00:51 - INFO - __main__ -     Num examples = 1043
07/09/2019 11:00:51 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 131/131 [00:17<00:00,  7.30it/s]
07/09/2019 11:01:09 - INFO - __main__ -   ***** Eval results *****
07/09/2019 11:01:09 - INFO - __main__ -     eval_loss = 0.4928024184555953
07/09/2019 11:01:09 - INFO - __main__ -     global_step = 804
07/09/2019 11:01:09 - INFO - __main__ -     loss = 0.09691035539949712
07/09/2019 11:01:09 - INFO - __main__ -     mcc = 0.5033071972797099
```