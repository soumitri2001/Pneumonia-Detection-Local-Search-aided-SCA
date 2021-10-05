# Pneumonia Detection using Local Search aided Sine-Cosine Algorithm
Based on our paper "Pneumonia Detection from Lung X-ray Images using Local Search Aided Sine Cosine Algorithm based Deep Feature Selection Method" accepted for publication in _International Journal of Intelligent Systems_, Wiley. DOI: https://doi.org/10.1002/int.22703 (will be available soon).

## Requirements
To install the required dependencies run the following in command prompt:
`pip install -r requirements.txt`

## Running the codes:
Required directory structure:

(Note: ``train`` and ``val`` contains subfolders representing classes in the dataset.)

```

+-- data
|   +-- .
|   +-- train
|   +-- val
+-- AbSCA.py
+-- local_search.py
+-- main.py

```
Then, run the code using the command prompt as follows:

`python main.py --data_directory "data"`

Available arguments:
- `--epochs`: Number of epochs of training. Default = 20
- `--learning_rate`: Learning Rate. Default = 0.001
- `--batch_size`: Batch Size. Default = 4
- `--momentum`: Momentum. Default = 0.9
