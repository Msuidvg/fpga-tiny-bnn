"""Run portable Python checks and, optionally, all four RTL parity benches."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.bnn.verify_bitexact import verify

def run(args,log_name):
    result=subprocess.run([str(x) for x in args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (ROOT/"build"/log_name).write_text(result.stdout,encoding="utf-8")
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)
    return result.stdout

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rtl",action="store_true",help="require Icarus Verilog on PATH")
    parser.add_argument("--bench",choices=[p.stem for p in sorted((ROOT/"sim").glob("tb_*.v"))],
                        help="run one RTL bench, for independent CI jobs")
    args=parser.parse_args()
    if args.bench and not args.rtl:parser.error("--bench requires --rtl")
    (ROOT/"build").mkdir(exist_ok=True)
    started=time.time()
    run([sys.executable,"-m","unittest","discover","-s","tools/bnn/tests","-v"],"python_tests.log")
    report=verify(ROOT/"data/gesture_v1.npz",ROOT/"models/v1/model.npz",None)
    if report["model"]["sha256"] != "12dd5fe1e6ae5402d0b08fe530f0a2937a59587bb6daebd8943f709c1d3c6717":
        raise SystemExit("Frozen V1 model hash mismatch")
    (ROOT/"build/bitexact_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if not report["passed"]: raise SystemExit("Python bit-exact verification failed")
    summary={"python_tests_passed":True,"bitexact_samples":report["dataset"]["samples"],
             "bitexact_passed":report["passed"],"rtl_requested":args.rtl,"rtl":{}}
    if args.rtl:
        iv=shutil.which("iverilog");vv=shutil.which("vvp")
        if not iv or not vv: raise SystemExit("Install iverilog and vvp, then retry --rtl")
        sources=sorted((ROOT/"rtl").glob("*.v"))
        for bench in sorted((ROOT/"sim").glob("tb_*.v")):
            if args.bench and bench.stem != args.bench:continue
            output=ROOT/"build"/(bench.stem+".vvp")
            run([iv,"-g2012","-DBNN_SIM_DEBUG","-I","testvectors/core","-I","testvectors/pipeline",
                 "-s",bench.stem,"-o",output,*sources,bench],bench.stem+"_compile.log")
            result=run([vv,output],bench.stem+".log")
            if "PASS" not in result or "FAIL" in result or "ERROR" in result:
                raise SystemExit("RTL did not report a clean PASS: "+bench.stem)
            summary["rtl"][bench.stem]={"passed":True,"last_line":result.strip().splitlines()[-1]}
            print(summary["rtl"][bench.stem]["last_line"],flush=True)
    summary["elapsed_seconds"]=round(time.time()-started,2)
    (ROOT/"build/verification_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
