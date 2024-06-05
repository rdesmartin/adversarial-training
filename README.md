# Property-driven Neural Network Training

This library is a generic implementation of adversarial training of neural networks using TensorFlow.

The goal is to train neural networks that have a stronger compliance to a desired user-defined property. This is performed using adversarial training, specifically using projected gradient descent (PGD) adversarial attacks.

Structure
------------
```
.
├── examples
│   └── NLP                                          - folder containing the NLP example
│       ├── data                                     - folder containing the data
│       ├── hyperrectangles                          - folder containing the hyper-rectangles
│       └── main.py                                  - file containing the main script to run the example
│   
└── src
    └── train.py                                     - file containing the train method
```
