# Food-101 ConvNeXt Reference Artifacts

These artifacts come from the reference run used in the CNNs Revisited chapter.
The run was executed on a CUDA reference workstation on 2026-04-27 with PyTorch 2.11.0+cu130,
TorchVision 0.26.0+cu130, CUDA, mixed precision, and a TorchVision
ConvNeXt-Tiny ImageNet-1K V1 checkpoint.

The committed files are the small provenance and reporting artifacts: command,
environment, metadata, metrics, history, confusion matrices, per-class
accuracy, prediction examples, and PNG plots. The selected checkpoint was saved
as `best_model.pt` on the remote run directory but is intentionally not
committed because it is a large binary artifact.

The prediction-example CSV files contain dataset paths, labels, predicted
labels, correctness flags, and confidences. They do not include copied
Food-101 photographs; image grids should be rendered from a local dataset cache
only when image-display rights and course policy permit that use.
