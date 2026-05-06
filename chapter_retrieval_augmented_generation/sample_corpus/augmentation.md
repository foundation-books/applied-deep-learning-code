# Data Augmentation

Data augmentation changes training examples in ways that should preserve the
label. In image classification, common augmentations include random crops,
horizontal flips, small rotations, color jitter, and mild blur. Augmentation is
usually applied to the training split, not to validation or test examples used
for model selection.

Good augmentation can improve validation accuracy because the model sees more
varied training inputs. Too much augmentation can hurt accuracy if the altered
images no longer match the task. A controlled experiment should change one
augmentation policy at a time and keep the validation set fixed.
