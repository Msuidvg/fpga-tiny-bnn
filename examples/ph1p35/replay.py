"""Prepare public replay fixtures and check packets from the example's UART."""

import argparse
import json
from pathlib import Path
import struct
import time


ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "testvectors" / "pipeline"
VECTOR_INDICES = (0, 79, 170)
EXPECTED_CYCLES = {"core": 103154, "pipeline": 257386}
MAGIC = b"BN"
PACKET = struct.Struct("<2sBBBB5iI")


def prepare(mode: str, output: Path) -> list[dict]:
    items = json.loads((VECTORS / "vectors.json").read_text(encoding="utf-8"))["items"]
    filename, size = ("expected_normalized.mem", 128) if mode == "core" else ("raw_masks.mem", 4608)
    words = (VECTORS / filename).read_text(encoding="ascii").split()
    payload = b"".join(int(words[index], 16).to_bytes(size, "little") for index in VECTOR_INDICES)
    cases = [
        {
            "case": case,
            "vector": index,
            "label": items[index]["label"],
            "class_id": items[index]["expected_model_label"],
            "score_margin": items[index]["confidence_margin"],
            "scores": items[index]["scores"],
            "cycles": EXPECTED_CYCLES[mode],
        }
        for case, index in enumerate(VECTOR_INDICES)
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples.mem").write_text("\n".join(f"{byte:02x}" for byte in payload) + "\n", encoding="ascii")
    (output / "expected.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    packets = b"".join(
        PACKET.pack(MAGIC, item["case"], 0, item["class_id"], item["score_margin"], *item["scores"], item["cycles"])
        for item in cases
    )
    (output / "expected_packets.mem").write_text("\n".join(f"{byte:02x}" for byte in packets) + "\n", encoding="ascii")
    return cases


def packet_matches(record: tuple, expected: list[dict]) -> bool:
    magic, case, status, class_id, margin, *scores_and_cycles = record
    if magic != MAGIC or status != 0 or case >= len(expected):
        return False
    item = expected[case]
    return (
        class_id == item["class_id"]
        and margin == item["score_margin"]
        and scores_and_cycles[:5] == item["scores"]
        and scores_and_cycles[-1] == item["cycles"]
    )


def check_serial(port: str, expected: list[dict], timeout: float) -> None:
    try:
        import serial
    except ImportError as error:
        raise SystemExit("Install pyserial with 'python -m pip install pyserial' for UART capture") from error
    seen = {}
    pending = bytearray()
    deadline = time.monotonic() + timeout
    with serial.Serial(port, 115200, timeout=0.1) as stream:
        while len(seen) < len(expected) and time.monotonic() < deadline:
            pending.extend(stream.read(1024))
            while len(pending) >= PACKET.size:
                start = pending.find(MAGIC)
                if start < 0:
                    del pending[:-1]
                    break
                del pending[:start]
                if len(pending) < PACKET.size:
                    break
                packet = PACKET.unpack_from(pending)
                if packet_matches(packet, expected):
                    seen[packet[1]] = {**expected[packet[1]], "cycles": packet[-1]}
                    del pending[:PACKET.size]
                else:
                    del pending[0]
    print(json.dumps({"received": sorted(seen.values(), key=lambda item: item["case"])}, indent=2))
    if len(seen) != len(expected):
        raise SystemExit(f"matched {len(seen)}/{len(expected)} UART cases before timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("core", "pipeline"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", help="optionally verify real UART output on this port")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    expected = prepare(args.mode, args.output)
    if args.port:
        check_serial(args.port, expected, args.timeout)


if __name__ == "__main__":
    main()
