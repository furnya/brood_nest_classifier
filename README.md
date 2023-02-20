# brood_nest_classifier

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
       [793.75, 693.75]]), ['(empty)', '(egg)', '(empty)', '(egg)', '(empty)', '(empty)', '(egg)', '(empty)'])
```