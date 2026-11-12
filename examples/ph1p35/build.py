"""Build one replay design in a fresh directory with Anlogic TD 6.2."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from replay import ROOT, prepare


HERE = Path(__file__).resolve().parent
TARGET = "PH1P35MDG324"


def write_project(output: Path, sources: list[str]) -> None:
    project = ET.Element("Project", Version="3", Minor="2", Path=output.as_posix())
    ET.SubElement(project, "TD_Version").text = "6.2.168116"
    ET.SubElement(project, "Name").text = "replay"
    hardware = ET.SubElement(project, "HardWare")
    for name, value in (("Family", "PH1"), ("Device", TARGET), ("Speed", "-3")):
        ET.SubElement(hardware, name).text = value

    files = ET.SubElement(project, "Source_Files")
    for tag, filenames, group in (
        ("Verilog", sources, "design_1"),
        ("ADC_FILE", ["pins.adc"], "constraint_1"),
        ("SDC_FILE", ["timing.sdc"], "constraint_1"),
    ):
        category = ET.SubElement(files, tag)
        for order, filename in enumerate(filenames, 2 if tag == "SDC_FILE" else 1):
            info = ET.SubElement(ET.SubElement(category, "File", Path=filename), "FileInfo")
            for name, value in (
                ("UsedInSyn", "true"), ("UsedInP&R", "true"),
                ("BelongTo", group), ("CompileOrder", str(order)),
            ):
                ET.SubElement(info, "Attr", Name=name, Val=value)
    sets = ET.SubElement(project, "FileSets")
    ET.SubElement(sets, "FileSet", Name="design_1", Type="DesignFiles")
    ET.SubElement(sets, "FileSet", Name="constraint_1", Type="ConstrainFiles")
    top = ET.SubElement(project, "TOP_MODULE")
    for name, value in (("LABEL", "replay_top"), ("MODULE", "replay_top"), ("CREATEINDEX", "user")):
        ET.SubElement(top, name).text = value
    ET.indent(project)
    ET.ElementTree(project).write(output / "replay.al", encoding="utf-8", xml_declaration=True)


def run_td(td: Path, output: Path, name: str, script: str) -> None:
    tcl = output / f"{name}.tcl"
    log = output / f"{name}.log"
    tcl.write_text(script, encoding="ascii")
    try:
        with log.open("wb") as stream:
            result = subprocess.run([str(td), tcl.name], cwd=output, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=900)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"TD {name} timed out. See {log}") from None
    content = log.read_text(encoding="utf-8", errors="replace")
    if result.returncode or f"TINY_BNN_{name.upper()}_DONE" not in content or re.search(r"\b(?:ERROR|FATAL):", content):
        raise SystemExit(f"TD {name} failed. See {log}")


def build(td: Path, mode: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    prepare(mode, output)
    for path in (ROOT / "models" / "v1" / "fpga").glob("*.mem"):
        shutil.copyfile(path, output / path.name)
    sources = sorted((ROOT / "rtl").glob("*.v"))
    for path in (*sources, HERE / "pins.adc", HERE / "timing.sdc"):
        shutil.copyfile(path, output / path.name)
    top = (HERE / "replay_top.v").read_text(encoding="ascii")
    if mode == "core":
        top = top.replace("parameter CORE_ONLY = 0,", "parameter CORE_ONLY = 1,", 1)
    (output / "replay_top.v").write_text(top, encoding="ascii")
    write_project(output, [path.name for path in sources] + ["replay_top.v"])
    run_td(td, output, "route", """import_device ph1_35p.db -package PH1P35MDG324 -speed 3
open_project replay.al -single_run
set_param rtl reg_threshold 0
set_param gate pack_seq_in_io off
elaborate -top replay_top
read_adc pins.adc
read_sdc timing.sdc
optimize_rtl
optimize_gate -packarea synthesis.area
legalize_phy_inst
place
route
report_area -io_info -file routed.area
update_timing -mode final
report_timing_summary -file final.timing
report_timing_status -file final.ts
export_db routed.db
puts TINY_BNN_ROUTE_DONE
exit
""")
    timing = (output / "final.timing").read_text(encoding="utf-8", errors="replace")
    fields = {}
    for name in ("SWNS", "STNS", "HWNS", "HTNS"):
        match = re.search(rf"\b{name}:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*ns", timing)
        if not match:
            raise SystemExit(f"missing {name} in {output / 'final.timing'}")
        fields[name] = float(match[1])
    if min(fields.values()) < 0 or "sys_clk_50m" not in timing:
        raise SystemExit(f"timing check failed: {fields}. See {output / 'final.timing'}")
    run_td(td, output, "bitgen", """import_device ph1_35p.db -package PH1P35MDG324 -speed 3
open_project replay.al -single_run
import_db routed.db
bitgen -bit replay.bit
puts TINY_BNN_BITGEN_DONE
exit
""")
    bitstream = output / "replay.bit"
    if not bitstream.is_file() or bitstream.stat().st_size == 0:
        raise SystemExit("TD did not produce a nonempty bitstream")
    print(json.dumps({"mode": mode, "timing_ns": fields, "bitstream": str(output / "replay.bit")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("core", "pipeline"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="new build directory")
    parser.add_argument("--td", default=os.environ.get("TD_COMMANDS_PROMPT") or shutil.which("td_commands_prompt"))
    args = parser.parse_args()
    if not args.td or not Path(args.td).is_file():
        parser.error("supply --td with the path to td_commands_prompt.exe")
    build(Path(args.td).resolve(), args.mode, args.output.resolve())


if __name__ == "__main__":
    main()
