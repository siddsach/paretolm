# Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the PTB dataset, provided.
The trained model can then be used by the generate script to generate new text.

The code is tested under Linux Anaconda 3 and reproducible.

```bash
python main.py         # Train a LSTM on PTB, reaching perplexity of 128.55 and take about 1 hour in slurm.ttic.edu. This is the baseline in this final project
python main.py --tied  # Train a tied LSTM on PTB
python main.py --epochs 40 --tied    # Train a tied LSTM on PTB for 40 epochs
python generate.py                      # Generate samples from the trained LSTM model.
```

Train log of command "python main.py":

| epoch   1 |   200/  829 batches | lr 20.00 | ms/batch 867.48 | loss  6.82 | ppl   912.32
| epoch   1 |   400/  829 batches | lr 20.00 | ms/batch 863.56 | loss  6.07 | ppl   434.53
| epoch   1 |   600/  829 batches | lr 20.00 | ms/batch 863.55 | loss  5.75 | ppl   313.20
| epoch   1 |   800/  829 batches | lr 20.00 | ms/batch 855.02 | loss  5.62 | ppl   274.76
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 738.51s | valid loss  5.49 | valid ppl   243.15
-----------------------------------------------------------------------------------------
| epoch   2 |   200/  829 batches | lr 20.00 | ms/batch 862.39 | loss  5.50 | ppl   243.52
| epoch   2 |   400/  829 batches | lr 20.00 | ms/batch 863.61 | loss  5.38 | ppl   216.14
| epoch   2 |   600/  829 batches | lr 20.00 | ms/batch 863.62 | loss  5.23 | ppl   187.72
| epoch   2 |   800/  829 batches | lr 20.00 | ms/batch 863.94 | loss  5.23 | ppl   186.19
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 740.42s | valid loss  5.16 | valid ppl   173.41
-----------------------------------------------------------------------------------------
| epoch   3 |   200/  829 batches | lr 20.00 | ms/batch 868.67 | loss  5.18 | ppl   177.20
| epoch   3 |   400/  829 batches | lr 20.00 | ms/batch 864.11 | loss  5.12 | ppl   167.62
| epoch   3 |   600/  829 batches | lr 20.00 | ms/batch 863.73 | loss  5.00 | ppl   147.95
| epoch   3 |   800/  829 batches | lr 20.00 | ms/batch 862.54 | loss  5.03 | ppl   152.43
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 741.54s | valid loss  5.02 | valid ppl   150.80
-----------------------------------------------------------------------------------------
| epoch   4 |   200/  829 batches | lr 20.00 | ms/batch 865.70 | loss  5.00 | ppl   148.52
| epoch   4 |   400/  829 batches | lr 20.00 | ms/batch 861.69 | loss  4.97 | ppl   144.52
| epoch   4 |   600/  829 batches | lr 20.00 | ms/batch 862.58 | loss  4.85 | ppl   127.68
| epoch   4 |   800/  829 batches | lr 20.00 | ms/batch 863.54 | loss  4.90 | ppl   134.72
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 740.39s | valid loss  4.94 | valid ppl   139.56
-----------------------------------------------------------------------------------------
| epoch   5 |   200/  829 batches | lr 20.00 | ms/batch 867.83 | loss  4.88 | ppl   132.06
| epoch   5 |   400/  829 batches | lr 20.00 | ms/batch 863.70 | loss  4.87 | ppl   130.04
| epoch   5 |   600/  829 batches | lr 20.00 | ms/batch 863.17 | loss  4.75 | ppl   115.27
| epoch   5 |   800/  829 batches | lr 20.00 | ms/batch 862.77 | loss  4.81 | ppl   122.72
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 741.20s | valid loss  4.89 | valid ppl   132.74




The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --decay DECAY      learning rate decay per epoch
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --log-interval N   report interval
  --save SAVE        path to save the final model
```

With these arguments, a variety of models can be tested.
[Recurrent Neural Network Regularization (Zaremba et al. 2014)](https://arxiv.org/pdf/1409.2329.pdf)
[Using the Output Embedding to Improve Language Models (Press & Wolf 2016](https://arxiv.org/abs/1608.05859)
[Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling (Inan et al. 2016)](https://arxiv.org/pdf/1611.01462.pdf)