"""Small metric, exact-image, ordering and review-presentation fixtures only."""
import copy,hashlib,json,re,shutil,struct,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import analyze as a
import metrics as m
import presentation as p

def image(value):return np.full((2,2,3),value,np.float64)

def tensor(path,value):
    data=np.ascontiguousarray(value,dtype='<f4').tobytes();header=json.dumps({'rgb':dict(dtype='F32',shape=list(value.shape),data_offsets=[0,len(data)])}).encode()
    path.write_bytes(struct.pack('<Q',len(header))+header+data);return hashlib.sha256(data).hexdigest()

def test_equal_and_known_pixel_error():
    z=image(0);assert m.score(z,z)==dict(mae_0_1=0.,mse_0_1=0.,rmse_0_1=0.,exact=True)
    b=z.copy();b[0,0,0]=1;score=m.score(b,z)
    assert score['mae_0_1']==1/12 and score['mse_0_1']==1/12 and score['rmse_0_1']==pytest.approx(np.sqrt(1/12))
    with pytest.raises(ValueError):m.score(z,np.zeros((2,2,4)))
    z[0,0,0]=np.nan
    with pytest.raises(ValueError):m.score(z,b)

def test_future_rmse_is_pooled_not_mean_rmse():
    value=m.average([m.score(image(0),image(0)),m.score(image(1),image(0))])
    assert value['mae_0_1']==.5 and value['mse_0_1']==.5 and value['rmse_0_1']==np.sqrt(.5)

def test_zero_stationary_baseline_and_reconstruction_drift():
    row=m.frame_scores(image(.25),image(0),image(0),image(.25),image(.25),image(0))
    assert row['original_repeat_vs_target']['exact'] and row['generated_change_from_own_first']['exact']
    assert row['generated_vs_target']['mae_0_1']==.25
    value=m.improvement(row['generated_vs_target'],row['original_repeat_vs_target'])
    assert value['strictly_lower_mae'] is None and value['relative_mse_improvement'] is None

def test_both_repeat_baselines_are_separate():
    row=m.frame_scores(image(.5),image(1),image(0),image(.25),image(.25),image(0))
    assert row['generated_vs_target']['mae_0_1']==.5
    assert row['original_repeat_vs_target']['mae_0_1']==1
    assert row['own_first_repeat_vs_target']['mae_0_1']==.75
    assert row['generated_change_from_previous']['mae_0_1']==.25
    assert m.improvement(row['generated_vs_target'],row['original_repeat_vs_target'])['strictly_lower_rmse']

def test_signed_pair_order_and_pair_drift():
    g={'closed':image(.25),'interact':image(.75)};first={'closed':image(.25),'interact':image(.25)}
    result=m.paired_scores(g,g,first);assert result['interact_minus_closed_error']['exact']
    assert result['paired_change_from_generated_first']['mae_0_1']==.5
    reverse={'closed':g['interact'],'interact':g['closed']}
    assert m.paired_scores(reverse,g,first)['interact_minus_closed_error']['rmse_0_1']==1

def test_raw_png_order_is_explicit_not_filesystem_order():
    index=dict(schema='wan22-rgb-frames-v1',purpose='generated_clip',shape=[1,3,17,2,2],dtype='float32',range=[-1,1],frames=[dict(index=i,file=f'{i:04d}.safetensors') for i in range(17)])
    assert len(a.frame_index(index,2,2))==17
    for change in ('swap','duplicate','padding','boolean'):
        bad=copy.deepcopy(index)
        if change=='swap':bad['frames'][1],bad['frames'][2]=bad['frames'][2],bad['frames'][1]
        elif change=='duplicate':bad['frames'][1]=bad['frames'][0]
        elif change=='padding':bad['frames'][0]['file']='0.safetensors'
        else:bad['frames'][0]['index']=False
        with pytest.raises(ValueError):a.frame_index(bad,2,2)

def test_generated_raw_to_png_exact_and_poison_rejected(tmp_path):
    root=tmp_path.resolve();raw=root/'decode/result/stationary_closed/rgb/0000.safetensors';raw.parent.mkdir(parents=True)
    value=(np.arange(12,dtype=np.float32).reshape(1,3,1,2,2)-6)/6;digest=tensor(raw,value)
    png=root/'decode/result/stationary_closed/frames/0000.png';png.parent.mkdir();Image.fromarray(a.raw._pixels(value)).save(png)
    class Reader:
        def take(self,name,expected,maximum):
            path=root/name;assert a.sha(path)==expected;return path
    outputs={'stationary_closed/rgb/0000.safetensors':a.sha(raw),'stationary_closed/frames/0000.png':a.sha(png)}
    index=[dict(index=0,file='0000.safetensors',sha256=a.sha(raw),bytes=raw.stat().st_size,tensor_sha256=digest)]
    result,_=a.generated_frame(Reader(),{'output_sha256':outputs},index,'stationary_closed',0,2,2)
    assert np.array_equal(result,(value[0,:,0].transpose(1,2,0).astype(np.float64)+1)/2)
    pixels=a.raw._pixels(value);pixels[0,0,0]^=1;Image.fromarray(pixels).save(png);outputs['stationary_closed/frames/0000.png']=a.sha(png)
    with pytest.raises(ValueError,match='rounding'):a.generated_frame(Reader(),{'output_sha256':outputs},index,'stationary_closed',0,2,2)

def test_target_opaque_alpha_and_exact_copy(tmp_path):
    root=tmp_path.resolve();path=root/'target.png';pixels=np.full((2,2,4),255,np.uint8);pixels[0,0,:3]=[0,127,128];Image.fromarray(pixels).save(path)
    assert np.array_equal(a.png_value(path,2,2,True),pixels[:,:,:3])
    out=root/'copy.png';a.copy_exact(path,out);assert path.read_bytes()==out.read_bytes()
    with pytest.raises(ValueError):a.copy_exact(path,out)
    pixels[0,0,3]=0;Image.fromarray(pixels).save(path)
    with pytest.raises(ValueError,match='Opaque'):a.png_value(path,2,2,True)

def test_fresh_output_and_incomplete_input_never_creates_results(tmp_path,monkeypatch):
    root=tmp_path.resolve();original=root/'original';original.mkdir();out=root/'output'
    with pytest.raises(ValueError):a.fresh_output(original,[original])
    with pytest.raises(ValueError):a.fresh_output(original/'nested',[original])
    monkeypatch.setattr(a,'cpu_review',lambda _: 'c'*64);monkeypatch.setattr(a,'Recovered',lambda *_:None)
    monkeypatch.setattr(a,'completed',lambda *_:(_ for _ in ()).throw(ValueError('incomplete actual run')))
    with pytest.raises(ValueError,match='incomplete'):a.analyze(original,original,original,out,root/'cpu.json')
    assert not out.exists()

def test_separate_unreviewed_forms_and_viewer_javascript(tmp_path):
    identity={'plan_sha256':'a'*64,'note':'</script>'};first=p.form(1,identity);second=p.form(2,identity)
    assert len(first['arms'])==6 and all(len(x['frames'])==17 for x in first['arms'].values())
    first['arms']['left_closed']['camera']='achieved';assert second['arms']['left_closed']['camera']=='unreviewed'
    root=tmp_path.resolve();p.viewer(root,identity,{a:{} for a in p.ARMS});html=(root/'index.html').read_text()
    assert html.count('</script>')==1 and '\\u003c/script>' in html
    script=re.search(r'<script>(.*)</script>',html,re.S).group(1);js=root/'viewer.js';js.write_text(script)
    node=shutil.which('node');assert node,'Installed Node required for bounded syntax check'
    subprocess.run([node,'--check',str(js)],capture_output=True,text=True,check=True,timeout=15)
    assert json.loads((root/'reviewer-1.json').read_text())['overall']=='unreviewed'
    assert 'Math.max(0' in script and 'max="16"' in html and "setTimeout" in script

def test_optional_mp4_uses_exact17_frame_sequence(tmp_path,monkeypatch):
    root=tmp_path.resolve();calls=[]
    monkeypatch.setattr(p.shutil,'which',lambda _: '/usr/bin/ffmpeg')
    def run(args,**kwargs):
        calls.append(args)
        if '-version' in args:return SimpleNamespace(stdout='ffmpeg fixture\n')
        Path(args[-1]).write_bytes(b'fixture');return SimpleNamespace(stdout='')
    monkeypatch.setattr(p.subprocess,'run',run);result=p.movies(root)
    assert result['status']=='completed' and len(result['clips'])==6
    for args in calls[1:]:
        assert args[args.index('-start_number')+1]=='0' and args[args.index('-frames:v')+1]=='17'
        assert args[args.index('-framerate')+1]=='25/3' and args[args.index('-i')+1].endswith('%04d.png')
    assert result['encoding'].startswith('Lossy')

def test_metric_scores_do_not_relabel_manual_outcomes():
    form=p.form(1,{'plan_sha256':'a'*64});perfect=m.score(image(0),image(0))
    assert perfect['exact'] and form['overall']=='unreviewed'
    assert all(r['camera']==r['door']=='unreviewed' for r in form['arms'].values())


def test_pair_viewer_waits_for_both_images_and_ignores_stale_load(tmp_path):
    root=tmp_path.resolve();p.viewer(root,{'plan_sha256':'a'*64},{a:{} for a in p.ARMS});html=(root/'index.html').read_text()
    helper=html.split('/*PAIR_BEGIN*/')[1].split('/*PAIR_END*/')[0]
    js=root/'synchronized.mjs'
    js.write_text("let displayRequest=0;const elements={mark:{},position:{},gen:{},target:{},genLink:{},targetLink:{}};const $=id=>elements[id];const pending={};globalThis.Image=class{set src(s){pending[s]=()=>this.onload()}};\n"+helper+"""
const first=showPair('left_closed',0);
pending['frames/target/left_closed/0000.png']();await Promise.resolve();
if(elements.gen.src||elements.target.src)throw Error('Half-loaded pair displayed');
pending['frames/generated/left_closed/0000.png']();await first;
if(!elements.gen.src.endsWith('/0000.png')||!elements.target.src.endsWith('/0000.png'))throw Error('First pair mismatch');
const stale=showPair('left_closed',1),fresh=showPair('right_interact',2);
pending['frames/generated/right_interact/0002.png']();pending['frames/target/right_interact/0002.png']();await fresh;
pending['frames/generated/left_closed/0001.png']();pending['frames/target/left_closed/0001.png']();await stale;
if(!elements.gen.src.endsWith('right_interact/0002.png')||!elements.target.src.endsWith('right_interact/0002.png')||elements.mark.disabled)throw Error('Stale pair replaced latest view');
""")
    subprocess.run([shutil.which('node'),str(js)],capture_output=True,text=True,check=True,timeout=15)
