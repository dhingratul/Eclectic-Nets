[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# LeNet
Implementation of LeNet using keras with tensorflow backend

# Requirements
1. python 2.7
2. tensorflow
3. keras

# Run
1. cd env/ and `conda env create -f environment.yml`
2. `source activate eclectic`
3. ``` python run.py --i "path to test image" ``` 
to produce prediction and display ground truth image
4. Note: You can use the following as a test case  ``` python run.py --i data/two.png ``` 

# Run Jupyter Notebook 
1. cd env/ and `conda env create -f environment.yml`
2. `source activate eclectic`
3. `jupyter notebook LeNet.ipynb` to run the jupyter notebook

# Note
1. ReLu activations has been used, not present in the original implementation
2. A comparison of 'sgd' vs 'adam' is done, final saved model uses 'adam'

# Acknowledgements
[SherlockLiao] 
[stared]
[jrosebr1]

