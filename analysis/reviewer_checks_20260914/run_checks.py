"""Read-only reassessment of existing predictions; no fitting or threshold selection."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import sys,json,csv,time,argparse,hashlib
from pathlib import Path
import numpy as np

ROOT=Path('/geniusland/home/huyaxuan/code/region_function_multimodal')
from quasf.decoders import decode, Prediction
from quasf.metrics import segmentation
from v7_joint_common9015 import load_regions,load_m2_probabilities,external_predictions,collapse_predictions,connected_components

def heterogeneous(labels,valid):
    rows=labels[np.asarray(valid,bool)]
    return len(rows)>=2 and len({tuple(np.flatnonzero(x)) for x in rows})>1

def full_truth(row):
    length=len(row['sequence']);masks=[];labels=[];valid=[]
    for d in row['domains']:
        mask=np.zeros(length,bool)
        for s,e in d['segments']:mask[max(0,s-1):min(length,e)]=True
        if not mask.any():continue
        lab=np.zeros(256,bool);lab[d.get('function_cluster_l2_indices',[])]=True
        masks.append(mask);labels.append(lab);valid.append(d.get('function_label_valid',False))
    return np.array(masks,bool).reshape(-1,length),np.array(labels,bool).reshape(-1,256),np.array(valid,bool)

def pad_predictions(preds,length):
    return [Prediction(p.query_id,np.pad(p.mask,(0,length-len(p.mask))),p.function_probs,p.mask_score,p.objectness) for p in preds]

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def collect_calls(pred,labels,valid,pairs):
    matched={i:j for i,j,_ in pairs};scores=[];targets=[]
    for i,p in enumerate(pred):
        j=matched.get(i)
        if j is not None and not valid[j]:continue
        scores.append(p.function_probs)
        targets.append(labels[j] if j is not None else np.zeros(labels.shape[1],bool))
    return np.asarray(scores,np.float32).reshape(-1,labels.shape[1]),np.asarray(targets,bool).reshape(-1,labels.shape[1])

def ranking(scores,targets,reference_total,outpath):
    score=scores.ravel();truth=targets.ravel();order=np.argsort(-score,kind='stable')
    s=score[order];y=truth[order];starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    ends=np.r_[starts[1:],len(s)];group_tp=np.add.reduceat(y.astype(np.int64),starts)
    cum=np.r_[0,np.cumsum(group_tp)];counts=ends-starts
    def at(k):
        k=min(int(k),len(s));idx=min(np.searchsorted(ends,k,side='left'),len(ends)-1)
        correct=float(cum[idx]+(k-starts[idx])*group_tp[idx]/counts[idx])
        return {'calls':k,'expected_correct':correct,'observed_precision':correct/k if k else 0.,'reference_recall':correct/reference_total if reference_total else 0.,'cutoff_score':float(s[starts[idx]]),'cutoff_tie_size':int(counts[idx])}
    budgets=sorted(set([1000,2000,4000,5606,7278,9149,12000,reference_total]))
    table=[at(k) for k in budgets if 0<k<=len(s)]
    curve=[at(k) for k in np.unique(np.linspace(1,min(len(s),20000),401,dtype=int))]
    write_csv(outpath,curve)
    return {'eligible_regions':len(scores),'candidate_calls':len(score),'reference_labels':reference_total,'budgets':table}

def summarize(rows):
    ans={}
    for group in ['all','untruncated','multidomain','discontinuous']:
      ans[group]={}
      for method in ['QuaSF','Chainsaw','Merizo']:
        r=[x for x in rows if x['method']==method and (group=='all' or x[group])]
        if not r:continue
        w=np.array([x['gt_residues'] for x in r]);tp=sum(x['domain_tp@0.5'] for x in r);fp=sum(x['domain_fp@0.5'] for x in r);fn=sum(x['domain_fn@0.5'] for x in r)
        ans[group][method]={'proteins':len(r),'domains':sum(x['gt_count'] for x in r),'predicted_domains':sum(x['pred_count'] for x in r),'weighted_iou':float(np.average([x['domain_iou'] for x in r],weights=w)),'mean_iou':float(np.mean([x['domain_iou'] for x in r])),'count_accuracy':float(np.mean([x['count_exact'] for x in r])),'weighted_boundary_mcc':float(np.average([x['boundary_mcc'] for x in r],weights=w)),'matched_domains_iou05':tp,'precision_iou05':tp/(tp+fp) if tp+fp else 0,'recall_iou05':tp/(tp+fn),'f1_iou05':2*tp/(2*tp+fp+fn)}
    rng=np.random.default_rng(20260914);cis={}
    for group in ['all','untruncated']:
      a=[x for x in rows if x['method']=='QuaSF' and (group=='all' or x[group])]
      for baseline in ['Chainsaw','Merizo']:
        b=[x for x in rows if x['method']==baseline and (group=='all' or x[group])]
        assert [x['accession'] for x in a]==[x['accession'] for x in b]
        w=np.array([x['gt_residues'] for x in a]);d=np.array([x['domain_iou']-y['domain_iou'] for x,y in zip(a,b)])
        dist=[]
        for _ in range(2000):
            ix=rng.integers(0,len(a),len(a));dist.append(np.average(d[ix],weights=w[ix]))
        cis[group+' vs '+baseline]={'difference':float(np.average(d,weights=w)),'ci95':np.quantile(dist,[.025,.975]).tolist()}
    return ans,cis

def main():
    global ROOT
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--chainsaw',type=Path,required=True);ap.add_argument('--merizo',type=Path,required=True);ap.add_argument('--m2-predictions',type=Path,required=True);ap.add_argument('--limit',type=int);args=ap.parse_args()
    ROOT=args.root
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cache=args.cache;protocol=json.loads((cache/'protocol.json').read_text());dataset=Path(protocol['signature']['dataset'])
    records={r['uniprot_acc']:r for r in map(json.loads,dataset.open())}
    index=json.loads((cache/'index.json').read_text());index=index[:args.limit] if args.limit else index
    # Baseline probability assets are external inputs, not obsolete source directories.
    baselines={'Chainsaw':load_regions(args.chainsaw),'Merizo':load_regions(args.merizo)}
    m2=load_m2_probabilities(args.m2_predictions)
    rows=[];hetrows=[];calls={p:{m:[[],[]] for m in ['QuaSF','Chainsaw','Merizo']} for p in ['components','identity']};totals={'components':0,'identity':0};reproduction={m:{'matched':0,'calls':0,'correct':0} for m in ['QuaSF','Chainsaw','Merizo']};length_audit=[];start=time.time();fullhet=0
    for number,entry in enumerate(index,1):
      with np.load(cache/entry['file'],allow_pickle=False) as sample:
        acc=str(sample['accession'].item());row=records[acc];length=len(row['sequence']);visible=int(sample['length']);gm,gl,gv=full_truth(row)
        cm=sample['gt_masks'].astype(bool);cl=sample['gt_labels'].astype(bool);cv=sample['gt_valid'].astype(bool)
        assert cm.shape[1]==visible
        if length==visible:
            assert np.array_equal(gm,cm),(acc,'masks')
            assert np.array_equal(gl,cl) and np.array_equal(gv,cv),(acc,'labels')
        else:length_audit.append({'accession':acc,'full_length':length,'visible_length':visible,'full_domains':len(gm),'visible_domains':len(cm)})
        native=decode(sample['mask_logits'],sample['func_logits'],sample['objectness_logits'],{'mode':'legacy_nms','mask_threshold':.3})
        pred={'QuaSF':pad_predictions(native,length)}
        for method,data in baselines.items():
            assert acc in data and acc in m2
            pred[method]=external_predictions(data[acc],length,m2[acc],Prediction)
        fullhet+=int(heterogeneous(gl,gv));discontinuous=any(len(d['segments'])>1 for d in row['domains'])
        identity_pairs={}
        for method,pp in pred.items():
            metrics,pairs=segmentation(pp,gm,length,thresholds=(.5,))
            rows.append({'accession':acc,'method':method,'untruncated':length==visible,'multidomain':len(gm)>1,'discontinuous':discontinuous,**metrics});identity_pairs[method]=pairs[.5]
        if heterogeneous(cl,cv):
            compm,compl,compv,contrib=connected_components(cm,cl,cv,visible)
            hetrows.append({'accession':acc,'untruncated':length==visible,'original_domains':len(cm),'original_annotated':int(cv.sum()),'components':len(compm),'annotated_components':int(compv.sum()),'heterogeneous_after':heterogeneous(compl,compv),'multi_source_components':sum(x>1 for x in contrib),'labels_original':int(cl[cv].sum()),'labels_components':int(compl[compv].sum())})
            totals['components']+=int(compl[compv].sum());totals['identity']+=int(gl[gv].sum())
            for method,pp in pred.items():
                visible_pred=native if method=='QuaSF' else external_predictions(baselines[method][acc],visible,m2[acc],Prediction)
                collapsed=collapse_predictions(visible_pred,visible,Prediction)
                _,pair=segmentation(collapsed,compm,visible,thresholds=(.5,))
                for version,pr,lab,val,match in [('components',collapsed,compl,compv,pair[.5]),('identity',pp,gl,gv,identity_pairs[method])]:
                    ss,tt=collect_calls(pr,lab,val,match);calls[version][method][0].append(ss);calls[version][method][1].append(tt)
                    if version=='components':
                        threshold={'QuaSF':.89,'Chainsaw':.31,'Merizo':.22}[method];keep=ss>=threshold
                        reproduction[method]['matched']+=len(match);reproduction[method]['calls']+=int(keep.sum());reproduction[method]['correct']+=int((keep&tt).sum())
        if number%250==0 or number==len(index):print(json.dumps({'done':number,'total':len(index),'seconds':round(time.time()-start,1),'heterogeneous':len(hetrows)}),flush=True)
    write_csv(out/'segmentation_per_protein.csv',rows);write_csv(out/'heterogeneity_per_protein.csv',hetrows);write_csv(out/'length_audit.csv',length_audit)
    seg,ci=summarize(rows);functional={}
    for version,by_method in calls.items():
      functional[version]={}
      for method,(ss,tt) in by_method.items():
        if not ss:continue
        s=np.concatenate(ss);t=np.concatenate(tt)
        np.savez_compressed(out/f'calls_{version}_{method}.npz',scores=s,targets=t)
        functional[version][method]=ranking(s,t,totals[version],out/f'curve_{version}_{method}.csv')
    result={'cohort':len(index),'segmentation':seg,'paired_bootstrap':ci,'length_audit':{'truncated_proteins':len(length_audit)},'heterogeneity':{'full_reference_heterogeneous':fullhet,'historical_heterogeneous':len(hetrows),'remaining_heterogeneous':sum(x['heterogeneous_after'] for x in hetrows),'collapsed_to_one_annotated_component':sum(x['annotated_components']==1 for x in hetrows),'multiple_components_identical_labels':sum(x['annotated_components']>=2 and not x['heterogeneous_after'] for x in hetrows),'original_domains':sum(x['original_domains'] for x in hetrows),'components':sum(x['components'] for x in hetrows),'annotated_components':sum(x['annotated_components'] for x in hetrows),'multi_source_components':sum(x['multi_source_components'] for x in hetrows),'reference_labels':totals},'table3_reproduction':reproduction,'functional_rankings':functional,'protocol':{'segmentation':'Complete original reference domain masks; official prediction identities retained; maximum-total-IoU matching; reference-length weights within proteins and GT-union length between proteins. No retuning. Full and untruncated cohorts reported.','functional':'Same historical heterogeneous cohort; reference identity and historical continuous components scored separately. IoU>=0.5 max cardinality then max total IoU. Unknown matched regions excluded; unmatched predictions count as calls without observed support. Same global top-K budget for each method; ties evaluated by expected correct count under uniform tie breaking.','uncertainty':'2000 paired protein bootstrap resamples for segmentation. Fixed-budget functional results are descriptive, not independently validated thresholds.','source_cache':protocol,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start}}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':True,'out':str(out),'reproduction':reproduction,'heterogeneity':result['heterogeneity']}),flush=True)

if __name__=='__main__':main()



