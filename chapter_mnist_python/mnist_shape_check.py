#!/usr/bin/env python3
"""Small shape diagnostic used by the MNIST Python chapter.

The chapter presents this as a habit to run after loading data. This standalone
version is deliberately dependency-free so that readers can run one example even
before installing Keras, PyTorch, or NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapePreview:
    shape: tuple[int, ...]
    dtype: str
    head: tuple[int, ...] = ()

    def __getitem__(self, item: slice) -> list[int]:
        if not isinstance(item, slice):
            raise TypeError("ShapePreview supports slices only")
        return list(self.head[item])


def main() -> int:
    x_train = ShapePreview(shape=(60000, 28, 28), dtype="uint8")
    y_train = ShapePreview(
        shape=(60000,),
        dtype="uint8",
        head=(5, 0, 4, 1, 9, 2, 1, 3, 1, 4),
    )

    print(x_train.shape)
    print(y_train.shape)
    print(x_train.dtype)
    print(y_train[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
