"""FPGA-oriented XNOR-popcount inference for exported Tiny-BNN models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


POPCOUNT8 = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class BitExactModel:
    class_labels: np.ndarray
    input_shape: tuple[int, int]
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    w1_packed: np.ndarray
    w2_packed: np.ndarray
    w3_packed: np.ndarray

    @property
    def architecture(self) -> tuple[int, int, int, int]:
        return (
            int(self.w1.shape[0]),
            int(self.w1.shape[1]),
            int(self.w2.shape[1]),
            int(self.w3.shape[1]),
        )


@dataclass(frozen=True)
class BitExactResult:
    predicted_labels: np.ndarray
    scores: np.ndarray
    h1_bits: np.ndarray
    h2_bits: np.ndarray
    h1_popcounts: np.ndarray
    h2_popcounts: np.ndarray
    output_popcounts: np.ndarray


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def activation_thresholds(feature_count: int, bias: np.ndarray) -> np.ndarray:
    """Minimum XNOR match count for signed_dot + bias >= 0."""

    values = feature_count - np.asarray(bias, dtype=np.int64)
    return (-np.floor_divide(-values, 2)).astype(np.int32)


def _expected_packed(weights: np.ndarray) -> np.ndarray:
    return np.packbits(
        (weights.T > 0).astype(np.uint8), axis=1, bitorder="little"
    )


def load_bitexact_model(path: Path) -> BitExactModel:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema_version",
            "class_labels",
            "input_shape",
            "w1",
            "b1",
            "w2",
            "b2",
            "w3",
            "b3",
            "w1_packed",
            "w2_packed",
            "w3_packed",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"model is missing arrays: {sorted(missing)}")
        if int(archive["schema_version"]) != 1:
            raise ValueError("unsupported model schema")
        arrays = {name: np.asarray(archive[name]).copy() for name in required}

    input_shape = tuple(int(value) for value in arrays["input_shape"])
    if len(input_shape) != 2 or any(value <= 0 for value in input_shape):
        raise ValueError(f"invalid input shape: {input_shape}")
    class_labels = arrays["class_labels"].astype(np.int8, copy=False)
    if class_labels.ndim != 1 or np.unique(class_labels).size != class_labels.size:
        raise ValueError("class labels must be a unique one-dimensional array")

    weights = [arrays[name] for name in ("w1", "w2", "w3")]
    biases = [arrays[name] for name in ("b1", "b2", "b3")]
    if any(weight.ndim != 2 for weight in weights):
        raise ValueError("all weight arrays must be matrices")
    if weights[0].shape[0] != int(np.prod(input_shape)):
        raise ValueError("input shape does not match first-layer weights")
    if weights[0].shape[1] != weights[1].shape[0]:
        raise ValueError("w1/w2 dimensions do not connect")
    if weights[1].shape[1] != weights[2].shape[0]:
        raise ValueError("w2/w3 dimensions do not connect")
    if weights[2].shape[1] != class_labels.size:
        raise ValueError("output dimension does not match class labels")
    for name, weight in zip(("w1", "w2", "w3"), weights):
        if weight.dtype != np.int8 or not np.all(np.isin(weight, (-1, 1))):
            raise ValueError(f"{name} must contain only int8 -1/+1 weights")
        if weight.shape[0] % 8:
            raise ValueError(f"{name} input width must be divisible by 8")
    for name, bias, weight in zip(("b1", "b2", "b3"), biases, weights):
        if bias.dtype != np.int32 or bias.shape != (weight.shape[1],):
            raise ValueError(f"{name} must be an int32 vector matching outputs")

    for name, weight in zip(("w1", "w2", "w3"), weights):
        packed = arrays[f"{name}_packed"]
        expected = _expected_packed(weight)
        if packed.dtype != np.uint8 or not np.array_equal(packed, expected):
            raise ValueError(f"{name}_packed does not match signed weights")

    return BitExactModel(
        class_labels=class_labels,
        input_shape=(input_shape[0], input_shape[1]),
        w1=weights[0],
        b1=biases[0],
        w2=weights[1],
        b2=biases[1],
        w3=weights[2],
        b3=biases[2],
        w1_packed=arrays["w1_packed"],
        w2_packed=arrays["w2_packed"],
        w3_packed=arrays["w3_packed"],
    )


def pack_activation_bits(values: np.ndarray, feature_count: int) -> np.ndarray:
    bits = (np.asarray(values).reshape(-1, feature_count) != 0).astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder="little")


def xnor_popcount(
    packed_inputs: np.ndarray,
    packed_weights: np.ndarray,
    feature_count: int,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    if feature_count % 8:
        raise ValueError("feature count must be divisible by 8")
    inputs = np.asarray(packed_inputs, dtype=np.uint8)
    weights = np.asarray(packed_weights, dtype=np.uint8)
    expected_bytes = feature_count // 8
    if inputs.ndim != 2 or inputs.shape[1] != expected_bytes:
        raise ValueError("packed input width does not match feature count")
    if weights.ndim != 2 or weights.shape[1] != expected_bytes:
        raise ValueError("packed weight width does not match feature count")
    output = np.empty((inputs.shape[0], weights.shape[0]), dtype=np.int32)
    for first in range(0, inputs.shape[0], chunk_size):
        last = min(first + chunk_size, inputs.shape[0])
        xor = np.bitwise_xor(inputs[first:last, None, :], weights[None, :, :])
        output[first:last] = POPCOUNT8[np.bitwise_not(xor)].sum(
            axis=2, dtype=np.int32
        )
    return output


def infer_bitexact(binary_inputs: np.ndarray, model: BitExactModel) -> BitExactResult:
    input_features, hidden1, hidden2, _ = model.architecture
    input_packed = pack_activation_bits(binary_inputs, input_features)
    h1_popcounts = xnor_popcount(input_packed, model.w1_packed, input_features)
    h1_bits = h1_popcounts >= activation_thresholds(input_features, model.b1)
    h1_packed = pack_activation_bits(h1_bits, hidden1)

    h2_popcounts = xnor_popcount(h1_packed, model.w2_packed, hidden1)
    h2_bits = h2_popcounts >= activation_thresholds(hidden1, model.b2)
    h2_packed = pack_activation_bits(h2_bits, hidden2)

    output_popcounts = xnor_popcount(h2_packed, model.w3_packed, hidden2)
    scores = (
        output_popcounts * 2
        - hidden2
        + model.b3.astype(np.int32, copy=False)
    )
    predicted_labels = model.class_labels[np.argmax(scores, axis=1)]
    return BitExactResult(
        predicted_labels=predicted_labels,
        scores=scores,
        h1_bits=h1_bits,
        h2_bits=h2_bits,
        h1_popcounts=h1_popcounts,
        h2_popcounts=h2_popcounts,
        output_popcounts=output_popcounts,
    )


def infer_signed_integer_reference(
    binary_inputs: np.ndarray, model: BitExactModel
) -> BitExactResult:
    input_features, hidden1, hidden2, _ = model.architecture
    input_bits = (
        np.asarray(binary_inputs).reshape(-1, input_features) != 0
    ).astype(np.int16)
    signed_inputs = input_bits * 2 - 1
    h1_pre = signed_inputs @ model.w1.astype(np.int16) + model.b1
    h1_bits = h1_pre >= 0
    signed_h1 = h1_bits.astype(np.int16) * 2 - 1
    h2_pre = signed_h1 @ model.w2.astype(np.int16) + model.b2
    h2_bits = h2_pre >= 0
    signed_h2 = h2_bits.astype(np.int16) * 2 - 1
    scores = signed_h2 @ model.w3.astype(np.int16) + model.b3
    predicted_labels = model.class_labels[np.argmax(scores, axis=1)]
    return BitExactResult(
        predicted_labels=predicted_labels,
        scores=scores.astype(np.int32, copy=False),
        h1_bits=h1_bits,
        h2_bits=h2_bits,
        h1_popcounts=((h1_pre - model.b1 + input_features) // 2).astype(np.int32),
        h2_popcounts=((h2_pre - model.b2 + hidden1) // 2).astype(np.int32),
        output_popcounts=((scores - model.b3 + hidden2) // 2).astype(np.int32),
    )
