# Pneumonia Detection using Local Search aided Sine-Cosine Algorithm
Based on our paper "Pneumonia Detection from Lung X-ray Images using Local Search Aided Sine Cosine Algorithm based Deep Feature Selection Method" under review in "International Journal of Intelligent Systems", Wiley.

# Requirements
To install the required dependencies run the following in command prompt:
`pip install -r requirements.txt`

# Running the codes:
Required directory structure:

(Note: Any number of classes can be present in the dataset. It will be captured by the code automatically)

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
