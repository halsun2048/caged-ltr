"""Formal R6.4 independent-dev quality, efficiency, calibration and CI evaluation."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer

def ndcg(values):
    values=list(values); top=values[:10]; dcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(top)); ideal=sorted(values,reverse=True)[:10]; idcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(ideal)); return float(dcg/idcg) if idcg else 0.
def metrics_for(scored, score):
    rows=[]; bucket_rows={b:[] for b in ('head','torso','tail')}; calibration=[]
    for query_id,group in scored.groupby('query_id'):
        ranked=group.sort_values(score,ascending=False).reset_index(drop=True); rel=ranked.graded_relevance.to_numpy(); relevant=np.flatnonzero(rel>0)
        rows.append({'query_id':query_id,'ndcg10':ndcg(rel),'hit10':float(np.any(rel[:10]>0)),'mrr':float(1/(relevant[0]+1)) if len(relevant) else 0.})
        raw=ranked[score].to_numpy(dtype=float); raw=np.clip(raw-raw.max(),-50,50); prob=np.exp(raw); prob/=prob.sum(); calibration.append((float(prob[0]),float(rel[0]>0)))
        for bucket in bucket_rows:
            mask=(ranked.frequency_bucket.to_numpy()==bucket); filtered=np.where(mask,rel,0.)
            if np.any(filtered>0): bucket_rows[bucket].append(ndcg(filtered))
    frame=pd.DataFrame(rows); confidence=np.array([x[0] for x in calibration]); correct=np.array([x[1] for x in calibration]); ece=0.
    for lo in np.linspace(0,1,11)[:-1]:
        hi=lo+.1; mask=(confidence>=lo)&(confidence<(hi if hi<1 else hi+1e-9));
        if mask.any(): ece+=mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return frame,{'ndcg10':float(frame.ndcg10.mean()),'hit10':float(frame.hit10.mean()),'mrr':float(frame.mrr.mean()),'buckets':{key:{'ndcg10':float(np.mean(values)),'queries':len(values)} for key,values in bucket_rows.items()},'calibration':{'ece10':float(ece),'brier_top1':float(np.mean((confidence-correct)**2))}}
def bootstrap(frames,iterations=2000):
    rng=np.random.default_rng(20240801); names=list(frames); n=len(next(iter(frames.values()))); samples={name:[] for name in names}; differences={name:[] for name in names if name!='tfidf'}
    arrays={name:frame.ndcg10.to_numpy() for name,frame in frames.items()}
    for _ in range(iterations):
        idx=rng.integers(0,n,n)
        for name in names: samples[name].append(float(arrays[name][idx].mean()))
        for name in differences: differences[name].append(float((arrays[name][idx]-arrays['tfidf'][idx]).mean()))
    return {'ndcg10_95ci':{name:[float(np.quantile(values,.025)),float(np.quantile(values,.975))] for name,values in samples.items()},'minus_tfidf_95ci':{name:[float(np.quantile(values,.025)),float(np.quantile(values,.975))] for name,values in differences.items()}}
class Encoder(torch.nn.Module):
    def __init__(self,name): super().__init__(); self.encoder=AutoModel.from_pretrained(name)
    def embed(self,batch):
        hidden=self.encoder(**batch).last_hidden_state; mask=batch['attention_mask'].unsqueeze(-1).to(hidden.dtype); return torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
def encoded(tok,texts,device,max_length): return {k:v.to(device) for k,v in tok(list(texts),padding=True,truncation=True,max_length=max_length,return_tensors='pt').items()}
@torch.inference_mode()
def score_encoder(checkpoint,name,tok,dev,device,batch_size,max_length):
    model=Encoder(name); state=torch.load(checkpoint,map_location='cpu'); model.load_state_dict(state['model'],strict=True); model.to(device).eval()
    warm=dev.head(min(batch_size,len(dev))); model.embed(encoded(tok,warm.passage,device,max_length));
    if device.type=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter(); query_rows=dev.groupby('query_id',sort=False).first().reset_index(); qvec={}
    for off in range(0,len(query_rows),batch_size):
        chunk=query_rows.iloc[off:off+batch_size]; vec=model.embed(encoded(tok,chunk["query"],device,max_length)).cpu(); qvec.update(zip(chunk.query_id,vec))
    scores=[]
    for off in range(0,len(dev),batch_size):
        chunk=dev.iloc[off:off+batch_size]; pvec=model.embed(encoded(tok,chunk.passage,device,max_length)).cpu(); qbatch=torch.stack([qvec[q] for q in chunk.query_id]); scores.extend((qbatch*pvec).sum(1).tolist())
    if device.type=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-started; peak=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0.; del model; torch.cuda.empty_cache() if device.type=='cuda' else None
    return np.array(scores),{'latency_ms_per_query':elapsed*1000/dev.query_id.nunique(),'throughput_queries_per_second':dev.query_id.nunique()/elapsed,'peak_gpu_mib':peak,'checkpoint_mib':Path(checkpoint).stat().st_size/2**20}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',type=Path,required=True); ap.add_argument('--dev',type=Path,required=True); ap.add_argument('--gate-routes',type=Path,required=True); ap.add_argument('--mind-checkpoint',type=Path,required=True); ap.add_argument('--distilled-checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--predictions',type=Path); ap.add_argument('--model',default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--max-length',type=int,default=128); args=ap.parse_args()
    train=pd.read_parquet(args.train); dev=pd.read_parquet(args.dev); routes=pd.read_parquet(args.gate_routes); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tok=AutoTokenizer.from_pretrained(args.model)
    dev['bm25']=-dev.bm25_rank.astype(float); dev['first']=dev.logit.astype(float); train_text=train['query']+' [SEP] '+train['passage']; dev_text=dev['query']+' [SEP] '+dev['passage']; vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True); x=vec.fit_transform(train_text); ridge=Ridge(alpha=1.0); ridge.fit(x,train.logit)
    start=time.perf_counter(); dev['tfidf']=ridge.predict(vec.transform(dev_text)); tfidf_elapsed=time.perf_counter()-start
    dev['mind_minilm'],mind_eff=score_encoder(args.mind_checkpoint,args.model,tok,dev,device,args.batch_size,args.max_length); dev['distilled_minilm'],distilled_eff=score_encoder(args.distilled_checkpoint,args.model,tok,dev,device,args.batch_size,args.max_length)
    route=routes.set_index('query_id'); use=dev.query_id.map(route.use_first).to_numpy(); dev['frozen_gate']=np.where(use,dev['first'],dev['bm25'])
    names=('bm25','tfidf','mind_minilm','distilled_minilm','first','frozen_gate'); frames={}; summary={}
    for name in names: frames[name],summary[name]=metrics_for(dev,name)
    train_q=train.groupby('query_id').first(); thresholds=np.quantile(train_q["query"].str.split().str.len(),[.33,.66]); query_lengths=dev.groupby('query_id').first()["query"].str.split().str.len(); query_bins=pd.cut(query_lengths,[-np.inf,thresholds[0],thresholds[1],np.inf],labels=['short','medium','long'],include_lowest=True); length_metrics={}
    for name,frame in frames.items():
        mapped=frame.set_index('query_id').join(query_bins.rename('length')); length_metrics[name]={bucket:float(group.ndcg10.mean()) for bucket,group in mapped.groupby('length',observed=True)}
    first_latency=float(routes.first_model_latency_ms.mean()); gate_latency=float(np.where(routes.use_first,routes.first_model_latency_ms,0.).mean())
    efficiency={'bm25':{'latency_ms_per_query':0.0,'latency_kind':'precomputed retrieval rank'},'tfidf':{'latency_ms_per_query':tfidf_elapsed*1000/dev.query_id.nunique(),'throughput_queries_per_second':dev.query_id.nunique()/tfidf_elapsed},'mind_minilm':mind_eff,'distilled_minilm':distilled_eff,'first':{'latency_ms_per_query':first_latency,'latency_kind':'recorded model prefill+decode'},'frozen_gate':{'latency_ms_per_query':gate_latency,'first_route_rate':float(routes.use_first.mean())},'first_cache_read':{'latency_kind':'I/O only; excluded from model comparison'}}
    payload={'schema':'nfcorpus_r6_formal_dev_v1','split':'independent_dev_only','queries':int(dev.query_id.nunique()),'models':summary,'query_length_ndcg10':length_metrics,'efficiency':efficiency,'bootstrap':bootstrap(frames),'test_accessed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    if args.predictions:
        args.predictions.parent.mkdir(parents=True,exist_ok=True); dev.to_parquet(args.predictions,index=False)
    print(json.dumps({'stage':'complete','output':str(args.output),'ndcg10':{k:v['ndcg10'] for k,v in summary.items()}}))
if __name__=='__main__': main()
