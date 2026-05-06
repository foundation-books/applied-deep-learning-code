# Validation Accuracy and Leakage

Validation accuracy is measured on a validation split that is not used for
gradient updates. It helps choose hyperparameters such as learning rate,
augmentation strength, model size, and training length. The final test set
should remain untouched until model selection is complete.

Test leakage happens when information from the test set influences training or
model selection. Repeatedly choosing a model because it improves test accuracy
turns the test set into another validation set. A careful report separates
training, validation, and test results and explains which split was used for
each decision.
