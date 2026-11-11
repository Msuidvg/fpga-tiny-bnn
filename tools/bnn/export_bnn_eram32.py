"""Derive 32-bit ERAM-friendly BNN weight memories from the frozen export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FPGA_DIR = (
    ROOT / "models" / "v1" / "fpga"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(
    source: Path,
    destination: Path,
    width: int,
    expected_rows: int,
) -> dict[str, object]:
    source_lines = source.read_text(encoding="ascii").splitlines()
    if width <= 0:
        raise ValueError(f"invalid feature width for {source.name}: {width}")
    if len(source_lines) != expected_rows:
        raise ValueError(
            f"{source.name} has {len(source_lines)} rows, expected {expected_rows}"
        )
    chunks_per_row = (width + 31) // 32
    chunk_lines: list[str] = []
    source_mask = (1 << width) - 1
    for row, line in enumerate(source_lines):
        value = int(line, 16)
        if value & ~source_mask:
            raise ValueError(
                f"{source.name} row {row} contains bits above feature width {width}"
            )
        chunk_lines.extend(
            f"{(value >> (32 * chunk)) & 0xFFFFFFFF:08X}"
            for chunk in range(chunks_per_row)
        )
    destination.write_text("\n".join(chunk_lines) + "\n", encoding="ascii")

    # Reassemble every neuron to prove that feature bit i is unchanged.
    for row, source_line in enumerate(source_lines):
        rebuilt = 0
        start = row * chunks_per_row
        for chunk in range(chunks_per_row):
            rebuilt |= int(chunk_lines[start + chunk], 16) << (32 * chunk)
        if rebuilt != (int(source_line, 16) & source_mask):
            raise ValueError(f"round-trip mismatch in {source.name} row {row}")

    return {
        "source": source.name,
        "source_sha256": sha256(source),
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "feature_bits": width,
        "neuron_rows": len(source_lines),
        "chunk_bits": 32,
        "chunks_per_neuron": chunks_per_row,
        "chunk_order": "neuron-major, least-significant/lowest-feature chunk first",
    }


def load_architecture(fpga_dir: Path) -> tuple[int, int, int, int]:
    manifest_path = fpga_dir / "bnn_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    architecture = manifest.get("architecture")
    if (
        not isinstance(architecture, list)
        or len(architecture) != 4
        or any(not isinstance(value, int) or value <= 0 for value in architecture)
    ):
        raise ValueError(
            f"invalid architecture in {manifest_path}: {architecture!r}"
        )
    return tuple(architecture)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fpga-dir", type=Path, default=DEFAULT_FPGA_DIR)
    args = parser.parse_args()
    fpga_dir = args.fpga_dir.resolve()
    input_bits, hidden1, hidden2, outputs = load_architecture(fpga_dir)
    files = [
        convert(
            fpga_dir / "bnn_w1.mem",
            fpga_dir / "bnn_w1_32.mem",
            input_bits,
            hidden1,
        ),
        convert(
            fpga_dir / "bnn_w2.mem",
            fpga_dir / "bnn_w2_32.mem",
            hidden1,
            hidden2,
        ),
        convert(
            fpga_dir / "bnn_w3.mem",
            fpga_dir / "bnn_w3_32.mem",
            hidden2,
            outputs,
        ),
    ]
    manifest = {
        "schema_version": 2,
        "purpose": "32-bit sequential ROM layout for PH1P ERAM inference",
        "source_export_manifest": "bnn_export_manifest.json",
        "architecture": [input_bits, hidden1, hidden2, outputs],
        "bit_exact_to_neuron_wide_export": True,
        "files": files,
    }
    manifest_path = fpga_dir / "bnn_eram32_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
