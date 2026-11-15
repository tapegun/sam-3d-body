"""Full pretrained SAM inference and matched SDF measurement comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import trimesh
from .measurements import measure, errors, calibrated, summarize

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def save(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False))
    temp.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='/home/tapegun/Projects/wg/data/anny_scale_v2')
    p.add_argument('--cache',default='/home/tapegun/Projects/wg/data/anny_cache_256')
    p.add_argument('--sdf-run',default='/home/tapegun/Projects/wg-sam3d-training/runs/sam3d_experiment')
    p.add_argument('--checkpoint',default='/home/tapegun/Projects/wg-sam3d-training/checkpoints/sam-3d-body-dinov3/model.ckpt')
    p.add_argument('--mhr',default='checkpoints/assets/mhr_model.pt')
    p.add_argument('--out',default='results/full_pretrained_test')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--bodies',type=int,default=128)
    p.add_argument('--wait-for-sdf',action='store_true')
    args=p.parse_args()
    if args.bodies < 1: p.error('bodies must be positive')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    save(out/'status.json',{'status':'waiting_for_sdf' if args.wait_for_sdf else 'starting'})
    if args.wait_for_sdf:
        stage=Path(args.sdf_run)/'stage.txt'
        while True:
            state=stage.read_text().strip()
            if state=='complete': break
            if state=='failed': raise RuntimeError('SDF pipeline failed; investigate before comparison')
            time.sleep(30)
    os.environ['MOMENTUM_ENABLED']='0'  # use official packaged TorchScript MHR
    import torch
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    data,cache=Path(args.data),Path(args.cache)
    meta=json.loads((cache/'cache.json').read_text())
    if meta['dataset_sha256'] != sha(data/'dataset.json'):
        raise ValueError('Dataset/cache mismatch')
    if meta['files']['images.npy']['sha256'] != sha(cache/'images.npy'):
        raise ValueError('Corrupt source images')
    images=np.load(cache/'images.npy',mmap_mode='r')
    ids=[(i,body) for i,body in enumerate(meta['ids'])
         if json.loads((data/body/'record.json').read_text())['split']==args.split][:args.bodies]
    if len(ids)!=args.bodies: raise ValueError('Insufficient bodies')
    model,cfg=load_sam_3d_body(args.checkpoint,mhr_path=args.mhr)
    model.requires_grad_(False)
    estimator=SAM3DBodyEstimator(model,cfg)
    report={'protocol':'fixed-height-section-proxies-v1','arguments':vars(args),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'checkpoint_sha256':sha(args.checkpoint),'mhr_sha256':sha(args.mhr),
        'dataset_sha256':meta['dataset_sha256'],'source_images_sha256':meta['files']['images.npy']['sha256'],
        'input':'same cached 256px front-view RGB as SDF; full-image crop; default perspective FOV',
        'limitation':'Synthetic orthographic images differ from SAM perspective training. Section proxies are not anatomical tape measurements.',
        'calibration':'Known GT height used only for explicitly labeled calibrated metrics, never inference.',
        'rows':[]}
    save(out/'status.json',{'status':'running','bodies':len(ids)})
    for index,body in ids:
        started=time.time()
        row={'body':body}
        gt=trimesh.load(data/body/'body.ply',force='mesh',process=False)
        truth=measure(gt)
        row['ground_truth']=truth
        try:
            image=np.array(images[index,0].transpose(1,2,0),copy=True)
            with torch.inference_mode():
                prediction=estimator.process_one_image(image,inference_type='full')[0]
            vertices=np.asarray(prediction['pred_vertices'],dtype=np.float64)
            np.savez_compressed(out/f'{body}_raw.npz',**prediction)
            # Official MHR head emits camera x-right/y-down/z-forward in meters.
            # Front-view dataset is x-right/y-up/z-back. No fitted alignment.
            vertices=vertices*np.array([1,-1,-1])
            mesh=trimesh.Trimesh(vertices,estimator.faces,process=False)
            mesh.export(out/f'{body}.ply')
            row['sam_native']=measure(mesh)
            row['sam_native_errors']=errors(row['sam_native'],truth)
            row['sam_height_calibrated_errors']=errors(measure(calibrated(mesh,gt.extents[1])),truth)
        except Exception as error:
            row['sam_failure']=f'{type(error).__name__}: {error}'
        sdf_path=Path(args.sdf_run)/'test'/body/'prediction.ply'
        if args.split=='test' and sdf_path.is_file():
            sdf=trimesh.load(sdf_path,force='mesh',process=False)
            row['sdf_native_errors']=errors(measure(sdf),truth)
            row['sdf_height_calibrated_errors']=errors(measure(calibrated(sdf,gt.extents[1])),truth)
        row['seconds']=time.time()-started
        report['rows'].append(row)
        for key in ['sam_native_errors','sam_height_calibrated_errors','sdf_native_errors','sdf_height_calibrated_errors']:
            report[key]=summarize(report['rows'],key)
        # Paired comparison avoids silently comparing different successful subsets.
        paired=[r for r in report['rows'] if 'sam_native_errors' in r and 'sdf_native_errors' in r]
        report['paired_bodies']=len(paired)
        report['paired']={k:summarize(paired,k) for k in ['sam_native_errors','sdf_native_errors']}
        save(out/'report.json',report)
        print(json.dumps({'body':body,'seconds':row['seconds'],'failure':row.get('sam_failure')}),flush=True)
        if len(report['rows'])==1 and 'sam_failure' in row:
            raise RuntimeError('First inference failed; stopping rather than repeating systemic failure')
    save(out/'status.json',{'status':'complete','bodies':len(ids),'sam_failures':sum('sam_failure' in r for r in report['rows'])})

if __name__=='__main__': main()
