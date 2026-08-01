"""Independent-dev comparison: FIRST vs TF-IDF vs MiniLM text student."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer

def ndcg(vals):
    vals=list(vals)[:10]; dcg=sum((2**v-1)/np.log2(i+2) for i,v in enumerate(vals)); ideal=sorted(vals,reverse=True); idcg=sum((2**v-1)/np.log2(i+2) for i,v in enumerate(ideal)); return float(dcg/idcg) if idcg else 0.
def split_ids(ids):
    order=sorted(ids,key=lambda x:hashlib.sha256(x.encode()).hexdigest()); return set(order[:int(.8*len(order))]),set(order[int(.8*len(order)):])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidates',required=True); ap.add_argument('--queries',required=True); ap.add_argument('--qrels',required=True); ap.add_argument('--labels',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--model',default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); ap.add_argument('--output',required=True); ap.add_argument('--max-length',type=int,default=128); ap.add_argument('--batch-size',type=int,default=64); args=ap.parse_args()
    cand=pd.read_parquet(args.candidates); queries=pd.read_parquet(args.queries).set_index('query_id')['query'].to_dict(); qrels=pd.read_parquet(args.qrels).set_index(['query_id','passage_id'])['graded_relevance'].to_dict(); labels=pd.read_parquet(args.labels); labels=labels[labels.variant=='baseline'];
    ids=sorted(cand.query_id.unique()); train_ids,dev_ids=split_ids(ids); cand['query_text']=cand.query_id.map(queries); cand['text']=cand.query_text+' [SEP] '+cand.passage
    # Item-frequency buckets are computed globally, with no qrels/test access.
    freq=cand.passage_id.value_counts(); qs=freq.quantile([.33,.66]).tolist();
    def bucket(pid): return 'tail' if freq.get(pid,0)<=qs[0] else ('torso' if freq.get(pid,0)<=qs[1] else 'head')
    # FIRST logits are the already generated baseline teacher scores.
    first=cand[['query_id','passage_id','bm25_rank']].copy(); lab=labels[['query_id','candidate_id','logit']].rename(columns={'candidate_id':'passage_id','logit':'first_score'}); first=first.merge(lab,on=['query_id','passage_id'],how='left'); first['first_score']=first.first_score.fillna(-1e9)
    # Fit TF-IDF only on the independent training queries and their teacher labels.
    tr=first[first.query_id.isin(train_ids)].merge(cand[['query_id','passage_id','text']],on=['query_id','passage_id']).dropna(subset=['first_score']); vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True); xtr=vec.fit_transform(tr.text); ridge=Ridge(alpha=1.0); ridge.fit(xtr,tr.first_score)
    dev=cand[cand.query_id.isin(dev_ids)].copy(); t0=time.perf_counter(); dev['tfidf_score']=ridge.predict(vec.transform(dev.text)); tfidf_ms=(time.perf_counter()-t0)*1000/len(dev_ids)
    # Load and score the real text encoder on GPU when available.
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tok=AutoTokenizer.from_pretrained(args.model); enc=AutoModel.from_pretrained(args.model).to(device).eval(); state=torch.load(args.checkpoint,map_location='cpu'); enc.load_state_dict({k[len('encoder.'):]:v for k,v in state['model'].items() if k.startswith('encoder.')},strict=False)
    @torch.inference_mode()
    def emb(texts):
        out=[]
        for i in range(0,len(texts),args.batch_size):
            b={k:v.to(device) for k,v in tok(texts[i:i+args.batch_size],padding=True,truncation=True,max_length=args.max_length,return_tensors='pt').items()}; h=enc(**b).last_hidden_state; m=b['attention_mask'].unsqueeze(-1).to(h.dtype); out.append(torch.nn.functional.normalize((h*m).sum(1)/m.sum(1).clamp_min(1),dim=-1).cpu())
        return torch.cat(out)
    qtexts=[queries[q] for q in dev.query_id]; ptexts=dev.passage.tolist(); t0=time.perf_counter(); qv=emb(qtexts); pv=emb(ptexts); dev['minilm_score']=(qv*pv).sum(1).numpy(); minilm_ms=(time.perf_counter()-t0)*1000/len(dev_ids)
    dev=dev.merge(first[['query_id','passage_id','first_score']],on=['query_id','passage_id']);
    def metric(model):
        rows=[]
        for q,g in dev.groupby('query_id'):
            s=g.sort_values(model,ascending=False); rel=[qrels.get((q,p),0) for p in s.passage_id]; rows.append({'query_id':q,'ndcg10':ndcg(rel),'bucket': 'all'})
            for b in ('head','torso','tail'):
                contrib=[(qrels.get((q,p),0) if bucket(p)==b else 0) for p in s.passage_id]; rows.append({'query_id':q,'ndcg10':ndcg(contrib),'bucket':b})
        f=pd.DataFrame(rows); return {'overall_ndcg10':float(f[f.bucket=='all'].ndcg10.mean()),'buckets':{b:float(f[f.bucket==b].ndcg10.mean()) for b in ('head','torso','tail')}}
    result={'schema':'mind_text_encoder_dev_comparison_v1','split':'independent_dev_20_percent','train_queries':len(train_ids),'dev_queries':len(dev_ids),'device':str(device),'models':{'first':metric('first_score'),'tfidf':metric('tfidf_score'),'minilm':metric('minilm_score')},'latency_ms_per_query':{'first':0.0,'tfidf':tfidf_ms,'minilm':minilm_ms},'frequency_bucket_thresholds':qs,'test_accessed':False,'tfidf_role':'efficiency_baseline'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); Path(args.output).with_suffix('.md').write_text('# Independent dev text comparison\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n'); print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
