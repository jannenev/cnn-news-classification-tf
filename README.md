
See modifications.txt

train2.py , without focus
train3.py , with focus

to run:
python ./train3.py --filter_sizes 3,5,7 --num_filters 256 --batch_size 64 --num_epochs 30

Trained models are large: 200-700 MB each, so for practical reasons they might not be included with code. 
With GTX 1070 GPU a model with 0.77 accuracy is reached by training about 5 minutes.

Path to pre-trained embedding file (for example "glove.6B.100d.txt") is defined in beginning of config.yml

### Modifying original Dennybritz code to multilabel 

In file data_helpers.py there is a function
```
load_data_and_labels()
```
, that loads data with binary-labels.

We added own function to load data with multi labels - in this case 5 labels.
```
def load_newsdata_and_labels():
```
In this function you read in your own data.
The labels need to be turned into one-hot form, ie if total 5 labels (0-4) and items label is 2, it would be: 0 0 1 0 0 (the bit 2 is 1 and rest 0)

So in the data_helpers.py function you load your own data, turn it to one-hot and return as: 
```
return [x_text, y]
```

Then in file train.py (in our code version train2.py) replace call to original data loading function with the own version.

```
# Load data
# was original
# print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

### Load own data ################
print("Loading data...")
x_text, y = data_helpers.load_newsdata_and_labels()
```



### From original Dennybritz version
https://github.com/dennybritz/cnn-text-classification-tf

## pre-trained embeddings can be downloaded from 
## https://nlp.stanford.edu/projects/glove/
## https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing  (https://code.google.com/archive/p/word2vec/)





**[This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)**

It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```bash
./train.py
```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
