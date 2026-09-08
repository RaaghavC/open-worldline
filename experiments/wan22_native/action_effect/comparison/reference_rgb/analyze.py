# SPDX-License-Identifier: Apache-2.0
"""CPU-only image correspondence diagnostics for the fixed start-0 paired clips."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / 'outputs/open-worldline'
sys.path.insert(0, str(REPO))
from experiments.wan22_native.spatial_audit import rgb as rgb_audit
from experiments.wan22_native.action_data import data as data_source
from experiments.wan22_native.spatial_reference.inputs import preprocess

ARMS = ('closed', 'open')
SELECTED = (0, 1, 2, 4, 8, 12, 16)
HEIGHT, WIDTH = 704, 1248


def sha(path):
    return rgb_audit._sha(Path(path))


def arr_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def sources():
    paths = [Path(__file__), Path(rgb_audit.__file__), Path(data_source.__file__),
             Path(data_source.original.__file__), Path(data_source.__file__).with_name('source-plan.md.txt'),
             REPO/'experiments/wan22_native/spatial_reference/inputs.py',
             REPO/'experiments/wan22_native/spatial_reference/config.py']
    return {str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p.relative_to(REPO)):sha(p) for p in paths}


def ground_truth(capture):
    """Load only original start-0 windows; derive the same canonical frame-0 RGB."""
    capture = Path(capture).resolve()
    require(sha(capture/'manifest.json') == data_source.MANIFEST_SHA256, 'Pinned original capture required')
    raw = {arm:data_source.original.load_window(capture, arm, 0, 17) for arm in ARMS}
    canonical = raw['open'].rgb[0]
    require(raw['open'].provenance['sources'][0]['sha256'] == data_source.CANONICAL_PNG_SHA256,
            'Canonical first PNG differs')
    windows = {arm:data_source.derive_window(raw[arm],canonical) for arm in ARMS}
    require(np.argwhere(windows['closed'].commands != windows['open'].commands).tolist() == [[0,5]],
            'Only outgoing initial interaction may differ')
    frames = {}; records = {}
    for arm in ARMS:
        frames[arm] = []
        for i,pixels in enumerate(windows[arm].rgb):
            processed, _, transform = preprocess(Image.fromarray(pixels, mode='RGB'), 'spatial')
            require(transform == {'resize':[1252,704], 'crop':[2,0,1250,704], 'resize_applied':True},
                    'Frozen spatial preprocessing differs')
            frames[arm].append(np.array(processed,dtype=np.uint8,copy=True))
        rows = []
        for i,pixels in enumerate(frames[arm]):
            rows.append({'generated_frame_index':i, 'raw_capture_frame':i,
                         'incoming_command':None if i==0 else ('wait' if arm=='closed' else 'interact') if i==1 else 'left',
                         'incoming_command_vector':None if i==0 else windows[arm].commands[i-1].tolist(),
                         'cumulative_requested_left_turn_degrees':max(0,i-1)*7.5,
                         'raw_png':raw[arm].provenance['sources'][i], 'processed_pixel_sha256':arr_sha(pixels)})
        records[arm] = {'commands_sha256':arr_sha(windows[arm].commands), 'frames':rows,
                        'derivation':windows[arm].provenance}
    require(np.array_equal(frames['closed'][0],frames['open'][0]), 'Derived starting RGB must match')
    return frames, {'capture_manifest_sha256':data_source.MANIFEST_SHA256,
        'preprocessing':transform, 'commands_channels':list(data_source.CHANNELS), 'arms':records,
        'canonicalization_changed_original_channel_values':18,'raw_files_modified':False}


def score(actual, target, mask=None):
    require(actual.shape==target.shape and actual.ndim==3 and actual.shape[-1]==3,
            'Equal HWC RGB arrays required')
    require(np.isfinite(actual).all() and np.isfinite(target).all(), 'Finite RGB required')
    delta=actual.astype(np.float64)-target.astype(np.float64)
    if mask is not None:
        require(mask.dtype==bool and mask.shape==actual.shape[:2], 'Spatial boolean mask required')
        if not mask.any():return None
        delta=delta[mask]
    mse=float(np.mean(delta*delta,dtype=np.float64))
    return {'mae_0_1':float(np.mean(np.abs(delta),dtype=np.float64)), 'mse_0_1':mse,
            'psnr_db':-10*math.log10(mse) if mse else None,'exact':mse==0}


def paired_frame(generated, truth):
    """No trained-quality threshold: report full-frame and paired-truth-region errors."""
    require(set(generated)==set(truth)==set(ARMS),'Both branches required')
    gt={arm:truth[arm].astype(np.float64)/255 for arm in ARMS}
    gen={arm:generated[arm].astype(np.float64) for arm in ARMS}
    byte_delta=np.abs(truth['open'].astype(np.int16)-truth['closed'].astype(np.int16))
    mask=np.max(byte_delta,axis=-1)>=2
    result={'truth_difference_mae_0_1':score(gt['open'],gt['closed'])['mae_0_1'],
            'generated_difference_mae_0_1':score(gen['open'],gen['closed'])['mae_0_1'],
            'truth_pixels_differing_any_byte':int(np.any(byte_delta!=0,axis=-1).sum()),
            'truth_pixels_differing_at_least_two_levels':int(mask.sum()),
            'difference_field_mae_0_1':float(np.mean(np.abs((gen['open']-gen['closed'])-(gt['open']-gt['closed'])),dtype=np.float64)),
            'mask_definition':'Any RGB channel differs by at least 2 uint8 levels between paired processed truth; includes lighting cues, not a door segmentation',
            'regions':{}}
    for label,region in [('full_frame',None),('paired_truth_difference',mask)]:
        arms={arm:{'own':score(gen[arm],gt[arm],region),
                   'opposite':score(gen[arm],gt['open' if arm=='closed' else 'closed'],region)} for arm in ARMS}
        informative=bool(np.any(byte_delta)) if region is None else bool(mask.any())
        correct={arm:(rows['own']['mae_0_1']<rows['opposite']['mae_0_1']) if rows['own'] else False for arm,rows in arms.items()}
        result['regions'][label]={'informative_truth_contrast':informative,'arms':arms,
            'strict_both_closer_to_own_truth':bool(informative and all(correct.values())),
            'ties_pass':False,'empty_region_scores':None}
    return result


def _completed(run):
    run=Path(run).resolve();parent=rgb_audit._json(rgb_audit._file(run,'metrics.json'))
    require(parent.get('status')=='passed' and parent.get('model_execution') is True
            and parent.get('frames_per_arm')==17 and parent.get('predictions')==200
            and parent.get('solver_updates')==100 and parent.get('model_frozen') is True,
            'Both complete frozen-model visual branches required')
    require(not list(run.rglob('watchdog-stop.json')), 'Stopped run is not a complete comparison')
    for stage in ('core','decode'):
        report_path=rgb_audit._file(run,stage+'/result/metrics.json')
        terminal_path=rgb_audit._file(run,stage+'/terminal.json')
        require(parent.get('child_reports',{}).get(stage)==sha(report_path) and
                parent.get('child_terminals',{}).get(stage)==sha(terminal_path),'Parent child hashes differ')
        terminal=rgb_audit._json(terminal_path)
        require(terminal.get('status')=='complete' and terminal.get('exit_code')==0 and not terminal.get('cleanup_error'),
                'Completed child terminal required')
        report=rgb_audit._json(report_path)
        require(report.get('status')=='passed' and report.get('stage')==stage
                and report.get('model_execution') is True and report.get('training') is False,
                'Completed inference-only stage required')
        require(report.get('plan_sha256')==parent.get('plan_sha256') and report.get('source_sha256')==parent.get('source_sha256'),
                'Stage plan or source identity differs')
    require(set(report.get('arms',{}))==set(ARMS) and
            all(row.get('cache_clear') is True and row.get('images',{}).get('frames')==17 for row in report['arms'].values()),
            'Both decoder arms and cleared cache required')
    return run,run/'decode/result',parent,report


def _bind_conditions(run,parent,provenance):
    plan_path=rgb_audit._file(run,'plan.json');plan=rgb_audit._json(plan_path)
    require(sha(plan_path)==parent.get('plan_sha256') and plan.get('profile')=='spatial'
            and plan.get('arms')==list(ARMS) and plan.get('same_initial_observation_noise_text') is True
            and plan.get('training') is False and plan.get('source_sha256')==parent.get('source_sha256'),
            'Matching same-input spatial visual plan required')
    commands_path=rgb_audit._file(run,'commands.safetensors',65536)
    require(sha(commands_path)==plan.get('artifacts',{}).get('commands.safetensors'),'Saved command file identity differs')
    commands=rgb_audit._TensorFile(commands_path,{arm:(1,16,6) for arm in ARMS},65536)
    core=rgb_audit._json(rgb_audit._file(run,'core/result/metrics.json'))
    rows={}
    for arm in ARMS:
        values,digest=commands.read(arm)
        require(digest==provenance['arms'][arm]['commands_sha256'], 'Generated commands differ from original destination commands')
        measured=core.get('arms',{}).get(arm,{})
        require(measured.get('commands_sha256')==digest and measured.get('predictions')==100
                and measured.get('solver_updates')==50, 'Measured command identity or counts differ')
        rows[arm]={'commands_sha256':digest,'input_tensor_sha256':measured.get('input_tensor_sha256'),
                   'text_tensor_sha256':measured.get('text_tensor_sha256')}
        require(isinstance(rows[arm]['input_tensor_sha256'],dict) and set(rows[arm]['input_tensor_sha256'])=={'initial_noise','initial_latent','observation','token_times'}
                and isinstance(rows[arm]['text_tensor_sha256'],dict) and set(rows[arm]['text_tensor_sha256'])=={'atrium','native_negative'},
                'Complete measured input and text tensor identities required')
        expected=plan.get('input_identity',{}).get('baseline',{})
        require(rows[arm]['input_tensor_sha256']==expected.get('input_tensor_sha256')
                and rows[arm]['text_tensor_sha256']==expected.get('context_tensor_sha256'),
                'Both sampled arms must retain the same exact baseline observation, noise and text identities')
    require(rows['closed']['input_tensor_sha256']==rows['open']['input_tensor_sha256']
            and rows['closed']['text_tensor_sha256']==rows['open']['text_tensor_sha256'], 'Matched initial inputs required')
    return {'plan_sha256':sha(plan_path),'commands_file_sha256':sha(commands_path),'arms':rows,
            'scope':'Input bytes and completed sampling records; no denoiser or solver replay'}


def _generated_indices(result,report):
    indices={}
    for arm in ARMS:
        path=rgb_audit._artifact(result,arm+'/rgb/index.json',report,rgb_audit.JSON_LIMIT)
        index=rgb_audit._json(path)
        require(index.get('schema')=='wan22-rgb-frames-v1' and index.get('purpose')=='generated_clip'
                and index.get('shape')==[1,3,17,HEIGHT,WIDTH] and index.get('dtype')=='float32'
                and index.get('range')==[-1,1] and len(index.get('frames',[]))==17,'Exact RGB index required')
        require({p.name for p in (result/arm/'rgb').iterdir()}=={'index.json'}|{f'{i:04d}.safetensors' for i in range(17)},
                'Complete exact raw frame files required')
        indices[arm]=index
    return indices


def _read_frame(result,report,index,arm,i):
    row=index['frames'][i];name=f'{i:04d}.safetensors'
    require(row.get('index')==i and row.get('file')==name,'Exact frame order required')
    path=rgb_audit._artifact(result,arm+'/rgb/'+name,report,rgb_audit.FRAME_LIMIT)
    require(row.get('sha256')==sha(path) and row.get('bytes')==path.stat().st_size,'Raw frame file hash differs')
    value,tensor_sha=rgb_audit._TensorFile(path,{'rgb':(1,3,1,HEIGHT,WIDTH)}).read('rgb')
    require(row.get('tensor_sha256')==tensor_sha and value.min()>=-1 and value.max()<=1,'Raw RGB identity/range differs')
    png_path=rgb_audit._artifact(result,arm+'/frames/'+f'{i:04d}.png',report,rgb_audit.IMAGE_LIMIT)
    pixels=rgb_audit._png(png_path,HEIGHT,WIDTH)
    require(np.array_equal(pixels,rgb_audit._pixels(value)),'Published frame differs from raw RGB conversion')
    return (value[0,:,0].transpose(1,2,0).astype(np.float64)+1)/2,pixels,{
        'raw_file_sha256':sha(path),'raw_tensor_sha256':tensor_sha,'png_sha256':sha(png_path),'png_pixels_sha256':arr_sha(pixels)}


def contact_sheet(selected,output):
    # Half-size nearest-neighbor display preserves aspect; no smoothing or enhancement.
    tw,th=624,352;label_height=42
    canvas=Image.new('RGB',(len(SELECTED)*tw,4*(th+label_height)),(242,242,238));draw=ImageDraw.Draw(canvas)
    for row,key in enumerate(('generated closed','generated open','truth closed','truth open')):
        for col,i in enumerate(SELECTED):
            x,y=col*tw,row*(th+label_height)
            action='initial image' if i==0 else ('wait' if key.endswith('closed') else 'interact') if i==1 else 'left'
            draw.text((x+8,y+5),f'{key} | frame {i} | incoming {action}',fill=(16,16,16))
            draw.text((x+8,y+21),'50% nearest-neighbor display; full PNGs remain unchanged',fill=(50,50,50))
            canvas.paste(Image.fromarray(selected[key][i]).resize((tw,th),Image.Resampling.NEAREST),(x,y+label_height))
    canvas.save(output)


def analyze(run,capture,output):
    output=Path(output);require(not output.exists(),'Fresh analysis output required')
    run,result,parent,report=_completed(run)
    truth,provenance=ground_truth(capture);conditions=_bind_conditions(run,parent,provenance)
    indices=_generated_indices(result,report)
    output.mkdir(parents=True)
    selected={key:{} for key in ('generated closed','generated open','truth closed','truth open')}
    initial=truth['open'][0].astype(np.float64)/255
    rows=[]
    for i in range(17):
        generated={};identities={};images={}
        for arm in ARMS:
            generated[arm],images[arm],identities[arm]=_read_frame(result,report,indices[arm],arm,i)
        row={'frame':i,'identities':identities,'pair':paired_frame(generated,{a:truth[a][i] for a in ARMS}),'arms':{}}
        for arm in ARMS:
            gt=truth[arm][i].astype(np.float64)/255
            row['arms'][arm]={'generated_vs_target':score(generated[arm],gt),'repeat_start_vs_target':score(initial,gt),
                'generated_vs_start':score(generated[arm],initial),'frame_mapping':provenance['arms'][arm]['frames'][i]}
            if i in SELECTED:
                selected['generated '+arm][i]=images[arm];selected['truth '+arm][i]=truth[arm][i]
        rows.append(row)
    contact_sheet(selected,output/'comparison.png')
    result_report={'schema':'worldline-action-visual-image-comparison-v1','status':'completed','model_execution':False,
        'generated_run_metrics_sha256':sha(run/'metrics.json'),'decoder_metrics_sha256':sha(result/'metrics.json'),
        'source_sha256':sources(),'data':provenance,'generated_conditions':conditions,'frames':rows,
        'future_frames_mean':{arm:{key:float(np.mean([x['arms'][arm][key]['mae_0_1'] for x in rows[1:]]))
                              for key in ('generated_vs_target','repeat_start_vs_target','generated_vs_start')} for arm in ARMS},
        'contact_frame_indices':list(SELECTED),'contact_sha256':sha(output/'comparison.png'),
        'all_frames_scored':17,'future_frames_scored':16,
        'meaning':'RGB correspondence only. No numerical threshold establishes visual quality or successful door control.',
        'limitations':['One seen fixed scene and one shared saved noise; no independent sample estimate.',
          'After the initial command there are 15 left turns, reaching 112.5 requested degrees at frame16. Door visibility is not measured by this RGB audit.',
          'Paired truth differences include indirect illumination and renderer variation, not just door pixels.',
          'A closer own-target score may reflect lighting or framing. Whole-frame target error is sensitive to camera mismatch.',
          'Frame0 is a conditioned reconstruction. It is reported separately from future averages.',
          'Both command sequences are supplied to positive and negative CFG branches; training used the positive context only.']}
    (output/'report.json').write_text(json.dumps(result_report,indent=2,allow_nan=False)+'\n')
    return result_report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=analyze(args.run,args.capture,args.output)
    print(json.dumps({'status':result['status'],'all_frames_scored':17,'model_execution':False}))


if __name__=='__main__':main()
