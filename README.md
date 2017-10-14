# Hackathon2017-ML
## Commands
### python train.py -h
```
usage: train.py [-h] [--data DATA] [--model MODEL] [--name NAME]

optional arguments:
  -h, --help     show this help message and exit
  --data DATA    Path with all the data to train on. (default:
                 data/dataset.csv)
  --model MODEL  Model to train with. [svm, naive_bayes, svm_grid_search]
                 (default: svm)
  --name NAME    Name of the training session. Used to save the model.
                 (default: my_model)
```

### python sample.py -h
```
usage: sample.py [-h] [--input INPUT] [--name NAME]

optional arguments:
  -h, --help     show this help message and exit
  --input INPUT  Data to classify (default: data/validation_manual/text_0.txt)
  --name NAME    Name of the training session. Used to save the model.
                 (default: my_model)
```
