# brood_nest_classifier

[![Documentation Status](https://readthedocs.org/projects/bn-classifier/badge/?version=latest)](http://bn-classifier.readthedocs.io/en/latest/?badge=latest)

This repository holds a classifier for brood nest backside scans.

Installation
================
pip install git+https://github.com/furnya/brood_nest_classifier.git


Usage
================

Usage example:
```
from bn_classifier.classifier import BNClassifier
image = "/path/to/image.png"
BNClassifier.get_prediction_points(image)
```

Output:
```
(array([[218.75, 193.75],
       [468.75, 193.75],
       [793.75, 218.75],
       [318.75, 443.75],
       [868.75, 493.75],
       [193.75, 668.75],
       [468.75, 693.75],
       [793.75, 693.75]]), [0, 1, 0, 1, 0, 0, 1, 0], ['(empty)', '(egg)'])
```

Data
================
The image data used for the training is located in the repository: https://github.com/furnya/bn_classifier_data
