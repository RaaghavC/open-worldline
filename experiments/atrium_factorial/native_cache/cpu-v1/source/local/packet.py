# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact native-resolution capture packet and source admission."""
from pathlib import Path
import shutil
from experiments.atrium_factorial import reader
from experiments.wan22_native.action_cuda import probe_evidence as evidence
from experiments.wan22_native.official_cpu.streaming import sha
from experiments.wan22_native.spatial_reference.guards import atomic, limits

HERE=Path(__file__).resolve().parent
REPO=evidence.REPO
SCHEMA='worldline-atrium-factorial-native-cache-v1'
SCOPE='six-arm-native-vae-cache-only'
MANIFEST='1872dea69d16b124fb7a0a9eefaff06ea2d850ff49f36bcf87fec8c10c82dada'
INDEX='ca29b3f1ac8f68d289b1d9fb54d3a4b15cf8d5d80dd41b917a4aa79dfcb8821a'
VALIDATION='1ce4b1bccab9f46f0b4843eb29a706401c4fb4d02a5290381e1f643a2d236e99'


def source_paths():
    result={'repo/'+name:path for name,path in evidence.source_paths().items()}
    for name in ('experiments/atrium_factorial/reader.py',
                 'experiments/atrium_factorial/validate.py',
                 'experiments/atrium_factorial/results/dataset-index.json',
                 'experiments/atrium_factorial/results/validation-v1.json',
                 'experiments/atrium_data/DATA-LICENSE','LICENSE',
                 'experiments/wan22_native/spatial_reference/guards.py',
                 'experiments/wan22_native/spatial_reference/__init__.py',
                 'experiments/wan22_native/cuda_reference/decode.py'):
        result['repo/'+name]=REPO/name
    for name in ('packet.py','cache.py','run.py','test_cpu.py'):
        result['local/'+name]=HERE/name
    return dict(sorted(result.items()))


def sources():return {k:sha(v) for k,v in source_paths().items()}


def protocol():
    return dict(scope=SCOPE,profile='spatial',arms=list(reader.ARMS),rgb_shape=[1,3,17,704,1248],
        target_shape=[1,48,5,44,78],observation_shape=[1,48,1,44,78],commands_shape=[1,16,6],
        encode_calls=8,one_frame_calls=2,seventeen_frame_calls=6,
        cross_length_checks=6,same_length_bit_exact_checks=5,repeated_image_bit_exact_checks=1,
        cross_length_limits={'max_abs':1e-5,'relative_l2':1e-5},
        pixel_operation='Published reader FP32 divide127.5 then subtract1; no resize, crop or canonicalization',
        commands_provided_to_codec=False,target_prefix_replaced=False,raw_capture_modified=False,
        independent_layouts=1,split='development',data_license='CC0-1.0',limits=limits('codec'),
        core_model_loaded=False,image_generation=False,quality_assessed=False,automatic_training_admission=False)


def input_plan(root):
    root=Path(root)
    index=REPO/'experiments/atrium_factorial/results/dataset-index.json'
    if sha(index)!=INDEX or sha(evidence.relative_file(root,'manifest.json'))!=MANIFEST:
        raise ValueError('Pinned published capture/index identity required')
    validation=REPO/'experiments/atrium_factorial/results/validation-v1.json'
    receipt=evidence.read_json(validation)
    if sha(validation)!=VALIDATION or receipt.get('status')!='passed' or receipt.get('manifest_sha256')!=MANIFEST:
        raise ValueError('Pinned passed full capture validation required')
    receipts={};inventory={'manifest.json':{'sha256':MANIFEST,'bytes':(root/'manifest.json').stat().st_size}}
    initial=None
    for arm in reader.ARMS:
        receipt=reader.verify_window(root,arm,expected_manifest_sha256=MANIFEST)
        if initial is None:initial=receipt['arrays']['initial_rgb']['sha256']
        if initial!=receipt['arrays']['initial_rgb']['sha256']:
            raise ValueError('All six original initial images must be exact')
        receipts[arm]=receipt
        inventory.update(receipt['images'])
    if len(inventory)!=103:raise ValueError('Exactly one manifest and102 PNG inputs required')
    for name,row in receipt['images'].items():
        if inventory.get(name)!={'sha256':row['sha256'],'bytes':row['bytes']}:
            raise ValueError('Capture inventory differs from passed independent validation')
    return dict(manifest_sha256=MANIFEST,public_index_sha256=INDEX,
                capture_validation_sha256=VALIDATION,
                public_release='https://github.com/RaaghavC/open-worldline/releases/tag/atrium-camera-door-factorial-v1',
                files=inventory,arms=receipts,initial_rgb_sha256=initial)


def review(path):
    result=evidence.read_json(path)
    if (result.get('status')!='passed' or result.get('tests',0)<5
            or result.get('source_sha256')!=sources() or result.get('sources_unchanged') is not True
            or any(type(result.get(k)) is not int or result[k]!=0 for k in ('pytest_exit_code','failures','errors','skipped'))):
        raise ValueError('Current complete source-bound CPU checks required')
    return sha(path)


def prepare(capture,cpu_report,output):
    root=Path(output)
    if root.exists() or root.is_symlink():raise ValueError('Fresh preparation directory required')
    cpu_hash=review(cpu_report);inputs=input_plan(capture);mapping=sources()
    root.mkdir(parents=True)
    for name,record in inputs['files'].items():
        path=root/'capture'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(evidence.relative_file(capture,name),path)
        if sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise ValueError('Copied capture changed')
    for name,path in source_paths().items():
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    shutil.copyfile(cpu_report,root/'cpu-report.json')
    plan=dict(schema=SCHEMA,status='prepared',model_execution=False,protocol=protocol(),
              input_plan=inputs,source_sha256=mapping,cpu_report_sha256=cpu_hash)
    atomic(root/'plan.json',plan);read_prepared(root)
    return plan


def read_prepared(root):
    root=Path(root);plan=evidence.read_json(evidence.relative_file(root,'plan.json'))
    if (plan.get('schema')!=SCHEMA or plan.get('status')!='prepared' or plan.get('protocol')!=protocol()
            or plan.get('source_sha256')!=sources()):raise ValueError('Prepared source or protocol changed')
    for name,digest in plan['source_sha256'].items():
        if sha(evidence.relative_file(root/'source',name))!=digest:raise ValueError('Source snapshot changed')
    if review(root/'cpu-report.json')!=plan['cpu_report_sha256']:raise ValueError('CPU evidence changed')
    if input_plan(root/'capture')!=plan['input_plan']:raise ValueError('Published RGB/action packet changed')
    return plan


def admission(path,root,plan,expected_gpu):
    value=evidence.read_json(path)
    required=dict(schema=SCHEMA,scope=SCOPE,decision='admit',plan_sha256=sha(Path(root)/'plan.json'),
        source_sha256=plan['source_sha256'],manifest_sha256=MANIFEST,cpu_report_sha256=plan['cpu_report_sha256'],
        expected_gpu=expected_gpu,limits=limits('codec'),core_model_loaded=False,training_admitted=False)
    if not expected_gpu or any(value.get(k)!=v for k,v in required.items()):
        raise ValueError('Separate source/input-bound parent VAE-cache admission required')
    return dict(sha256=sha(path),**required)
