"""Experimental all-NumPy convolutional BNN for 32x32 binary hand masks."""

from __future__ import annotations

import math

import numpy as np

try:
    from .train_bnn import Adam, _augment, _binary_sign
except ImportError:
    from train_bnn import Adam, _augment, _binary_sign


def _im2col(inputs: np.ndarray, kernel: int) -> tuple[np.ndarray, int, int]:
    windows = np.lib.stride_tricks.sliding_window_view(
        inputs, (kernel, kernel), axis=(2, 3)
    )
    out_h, out_w = windows.shape[2:4]
    columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
        -1, inputs.shape[1] * kernel * kernel
    )
    return columns, out_h, out_w


def _conv_forward(
    inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, tuple[object, ...]]:
    kernel = weights.shape[2]
    columns, out_h, out_w = _im2col(inputs, kernel)
    binary_weights = _binary_sign(weights)
    scale = math.sqrt(weights.shape[1] * kernel * kernel)
    output = columns @ binary_weights.reshape(weights.shape[0], -1).T
    output = output / scale + bias
    output = output.reshape(inputs.shape[0], out_h, out_w, weights.shape[0])
    output = output.transpose(0, 3, 1, 2)
    return output, (inputs.shape, columns, binary_weights, out_h, out_w, scale)


def _conv_backward(
    gradient: np.ndarray, cache: tuple[object, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_shape, columns, binary_weights, out_h, out_w, scale = cache
    input_shape = tuple(int(value) for value in input_shape)
    binary_weights = np.asarray(binary_weights)
    batch, channels, height, width = input_shape
    filters, _, kernel, _ = binary_weights.shape
    matrix_gradient = gradient.transpose(0, 2, 3, 1).reshape(-1, filters)
    weight_gradient = (matrix_gradient.T @ columns) / float(scale)
    weight_gradient = weight_gradient.reshape(binary_weights.shape)
    bias_gradient = matrix_gradient.sum(axis=0)
    column_gradient = (
        matrix_gradient @ binary_weights.reshape(filters, -1)
    ) / float(scale)
    column_gradient = column_gradient.reshape(
        batch, int(out_h), int(out_w), channels, kernel, kernel
    )
    input_gradient = np.zeros(input_shape, dtype=np.float32)
    for ky in range(kernel):
        for kx in range(kernel):
            contribution = column_gradient[:, :, :, :, ky, kx].transpose(
                0, 3, 1, 2
            )
            input_gradient[
                :, :, ky : ky + int(out_h), kx : kx + int(out_w)
            ] += contribution
    return input_gradient, weight_gradient, bias_gradient


class BinaryConvNet:
    """32x32 -> Conv5x5x8 -> majority pool2 -> Conv3x3x16 -> Dense5."""

    def __init__(
        self, classes: int, rng: np.random.Generator, pool_size: int = 2
    ) -> None:
        if pool_size not in (2, 4):
            raise ValueError("pool_size must be 2 or 4")
        self.pool_size = pool_size
        pooled_side = 28 // pool_size
        conv2_side = pooled_side - 2
        self.w1 = rng.normal(0.0, 0.12, (8, 1, 5, 5)).astype(np.float32)
        self.b1 = np.zeros(8, dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.10, (16, 8, 3, 3)).astype(np.float32)
        self.b2 = np.zeros(16, dtype=np.float32)
        self.w3 = rng.normal(
            0.0, 0.08, (16 * conv2_side * conv2_side, classes)
        ).astype(
            np.float32
        )
        self.b3 = np.zeros(classes, dtype=np.float32)

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
        }

    def forward(
        self, flat_inputs: np.ndarray
    ) -> tuple[np.ndarray, tuple[object, ...]]:
        inputs = np.asarray(flat_inputs, dtype=np.float32).reshape(-1, 1, 32, 32)
        z1, conv1_cache = _conv_forward(inputs, self.w1, self.b1)
        h1 = _binary_sign(z1)
        pooled_side = 28 // self.pool_size
        pool_raw = h1.reshape(
            -1, 8, pooled_side, self.pool_size, pooled_side, self.pool_size
        ).mean(axis=(3, 5))
        pool = _binary_sign(pool_raw)
        z2, conv2_cache = _conv_forward(pool, self.w2, self.b2)
        h2 = _binary_sign(z2)
        flattened = h2.reshape(h2.shape[0], -1)
        bw3 = _binary_sign(self.w3)
        dense_scale = math.sqrt(self.w3.shape[0])
        logits = (flattened @ bw3) / dense_scale + self.b3
        cache = (
            z1,
            h1,
            pool_raw,
            z2,
            h2,
            flattened,
            bw3,
            conv1_cache,
            conv2_cache,
            dense_scale,
        )
        return logits, cache

    def loss_and_gradients(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        class_weights: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self.forward(inputs)
        (
            z1,
            h1,
            pool_raw,
            z2,
            h2,
            flattened,
            bw3,
            conv1_cache,
            conv2_cache,
            dense_scale,
        ) = cache
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        sample_weights = class_weights[targets].astype(np.float32)
        sample_weights /= max(float(sample_weights.mean()), 1e-8)
        loss = -float(
            np.mean(
                sample_weights
                * np.log(probabilities[np.arange(targets.size), targets] + 1e-9)
            )
        )
        dlogits = probabilities
        dlogits[np.arange(targets.size), targets] -= 1.0
        dlogits *= sample_weights[:, None] / targets.size

        gradients: dict[str, np.ndarray] = {}
        gradients["w3"] = (flattened.T @ dlogits) / dense_scale
        gradients["b3"] = dlogits.sum(axis=0)
        dh2 = (dlogits @ bw3.T).reshape(h2.shape) / dense_scale
        dz2 = dh2 * (np.abs(z2) <= 1.0)
        dpool, gradients["w2"], gradients["b2"] = _conv_backward(
            dz2, conv2_cache
        )
        dpool_raw = dpool * (np.abs(pool_raw) <= 1.0)
        pool_area = float(self.pool_size * self.pool_size)
        dh1 = np.repeat(
            np.repeat(dpool_raw / pool_area, self.pool_size, axis=2),
            self.pool_size,
            axis=3,
        )
        dz1 = dh1 * (np.abs(z1) <= 1.0)
        _, gradients["w1"], gradients["b1"] = _conv_backward(dz1, conv1_cache)
        for name in ("w1", "w2", "w3"):
            gradients[name] *= np.abs(self.parameters[name]) <= 1.0
        return loss, gradients

    def deployment_arrays(self) -> dict[str, np.ndarray]:
        return {
            "w1": _binary_sign(self.w1).astype(np.int8),
            "b1": np.rint(self.b1 * math.sqrt(25)).astype(np.int32),
            "w2": _binary_sign(self.w2).astype(np.int8),
            "b2": np.rint(self.b2 * math.sqrt(8 * 9)).astype(np.int32),
            "w3": _binary_sign(self.w3).astype(np.int8),
            "b3": np.rint(self.b3 * math.sqrt(self.w3.shape[0])).astype(
                np.int32
            ),
            "pool_size": np.asarray(self.pool_size, dtype=np.int8),
        }


def _integer_conv(
    inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    columns, out_h, out_w = _im2col(inputs, weights.shape[2])
    output = columns.astype(np.int16) @ weights.reshape(weights.shape[0], -1).T
    output = output.astype(np.int32) + bias
    return output.reshape(inputs.shape[0], out_h, out_w, weights.shape[0]).transpose(
        0, 3, 1, 2
    )


def predict_deployment(
    inputs: np.ndarray, arrays: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(inputs, dtype=np.int8).reshape(-1, 1, 32, 32)
    h1 = np.where(_integer_conv(image, arrays["w1"], arrays["b1"]) >= 0, 1, -1)
    pool_size = int(arrays["pool_size"])
    pooled_side = 28 // pool_size
    pooled = np.where(
        h1.reshape(
            -1, 8, pooled_side, pool_size, pooled_side, pool_size
        ).sum(axis=(3, 5))
        >= 0,
        1,
        -1,
    )
    h2 = np.where(
        _integer_conv(pooled, arrays["w2"], arrays["b2"]) >= 0, 1, -1
    )
    scores = h2.reshape(h2.shape[0], -1) @ arrays["w3"].astype(np.int16)
    scores = scores.astype(np.int32) + arrays["b3"]
    return np.argmax(scores, axis=1), scores


def train_model(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int = 160,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    seed: int = 20260930,
    max_shift: int = 1,
    noise_probability: float = 0.002,
    verbose: bool = True,
) -> BinaryConvNet:
    rng = np.random.default_rng(seed)
    class_count = int(targets.max()) + 1
    model = BinaryConvNet(class_count, rng, pool_size=2)
    optimizer = Adam(model.parameters)
    counts = np.bincount(targets, minlength=class_count).astype(np.float32)
    class_weights = targets.size / (class_count * counts)
    for epoch in range(epochs):
        order = rng.permutation(targets.size)
        losses: list[float] = []
        epoch_lr = learning_rate * (
            0.15 + 0.85 * (1.0 + math.cos(math.pi * epoch / epochs)) / 2.0
        )
        for start in range(0, targets.size, batch_size):
            indices = order[start : start + batch_size]
            batch = _augment(
                inputs[indices],
                rng,
                max_shift=max_shift,
                noise_probability=noise_probability,
            )
            loss, gradients = model.loss_and_gradients(
                batch, targets[indices], class_weights
            )
            optimizer.step(model.parameters, gradients, epoch_lr)
            losses.append(loss)
        if verbose and (
            epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs
        ):
            predicted, _ = predict_deployment(inputs, model.deployment_arrays())
            print(
                f"conv_epoch={epoch + 1:03d}/{epochs} "
                f"loss={np.mean(losses):.4f} "
                f"deploy_train={np.mean(predicted == targets):.4f}"
            )
    return model
