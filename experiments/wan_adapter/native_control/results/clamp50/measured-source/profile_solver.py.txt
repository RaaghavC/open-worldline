# SPDX-License-Identifier: Apache-2.0
"""Measure 50 CPU solver steps and MPS transfers using a constant oracle.

No learned network or external weights are loaded. Tensor outputs are numerical
solver tests and are not decoded or presented as generated videos.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from fetch_weights import sha
from native_control.loop import integrate
from native_control.sampling import initial_noise
from native_control.evidence import Evidence


class ConstantOracle:
    def __call__(self, values, timesteps, contexts, length):
        return [torch.full_like(value,.2) for value in values]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    evidence=Evidence(args.output)
    report=evidence.report
    error=None
    try:
        noise=initial_noise();context=torch.zeros(1,4096)
        records=[]
        def record(index,timestep,latent,times):
            records.append(times)
        result=evidence.measure('50_steps_cpu_solver_mps_transfers',lambda:integrate(ConstantOracle(),noise,context,context,device='mps',callback=record))
        reference=evidence.measure('50_steps_all_cpu_reference',lambda:integrate(ConstantOracle(),noise,context,context,device='cpu'))
        difference=(result-reference).abs()
        if not torch.equal(result,reference):
            raise RuntimeError('CPU solver plus MPS transfer changed oracle result')
        report.update(purpose='Actual-shape CPU UniPC and transfer overhead with constant oracle, no network loaded',
                      steps=50,model_calls=100,loop_sha256=sha(Path(__file__).with_name('loop.py')),
                      source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('sampling.py'),Path(__file__).with_name('loop.py'),Path(__file__).with_name('evidence.py'))},
                      oracle_cpu_max_abs=float(difference.max()),
                      input_transfers_seconds=sum(row['input_transfer_seconds'] for row in records),
                      output_transfers_seconds=sum(row['output_transfer_seconds'] for row in records),
                      cpu_solver_seconds=sum(row['cpu_solver_seconds'] for row in records))
        report['solver_and_transfers_seconds']=sum(report[key] for key in ('input_transfers_seconds','output_transfers_seconds','cpu_solver_seconds'))
    except BaseException as exc:
        error=exc
        raise
    finally:
        evidence.close(error)
    print(report)


if __name__=='__main__':main()
