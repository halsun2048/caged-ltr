"""GPU-score locked-test inputs without reading qrels."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import pandas as pd,torch
from transformers import AutoModel,AutoTokenizer
class Encoder(torch.nn.Module):
    def __init__(self,name): super().__init__(); self.encoder=AutoModel.from_pretrained(name)
    def embed(self,batch):
        hidden=self.encoder(**batch).last_hidden_state; mask=batch['attention_mask'].unsqueeze(-1).to(hidden.dtype); return torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
def encoded(tok,texts,device,length): return {k:v.to(device) for k,v in tok(list(texts),padding=True,truncation=True,max_length=length,return_tensors='pt').items()}
@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--metadata',type=Path,required=True); ap.add_argument('--model',default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--max-length',type=int,default=128); args=ap.parse_args(); frame=pd.read_parquet(args.inputs); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tok=AutoTokenizer.from_pretrained(args.model); model=Encoder(args.model); model.load_state_dict(torch.load(args.checkpoint,map_location='cpu')['model'],strict=True); model.to(device).eval(); warm=frame.head(args.batch_size); model.embed(encoded(tok,warm.passage,device,args.max_length)); torch.cuda.synchronize() if device.type=='cuda' else None; started=time.perf_counter(); query_rows=frame.groupby('query_id',sort=False).first().reset_index(); qvec={}
    for off in range(0,len(query_rows),args.batch_size):
        chunk=query_rows.iloc[off:off+args.batch_size]; vec=model.embed(encoded(tok,chunk['query'],device,args.max_length)).cpu(); qvec.update(zip(chunk.query_id,vec))
    scores=[]
    for off in range(0,len(frame),args.batch_size):
        chunk=frame.iloc[off:off+args.batch_size]; pvec=model.embed(encoded(tok,chunk.passage,device,args.max_length)).cpu(); scores.extend((torch.stack([qvec[q] for q in chunk.query_id])*pvec).sum(1).tolist())
    torch.cuda.synchronize() if device.type=='cuda' else None; elapsed=time.perf_counter()-started; frame['distilled_minilm']=scores; args.output.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(args.output,index=False); args.metadata.write_text(json.dumps({'queries':int(frame.query_id.nunique()),'rows':len(frame),'latency_ms_per_query':elapsed*1000/frame.query_id.nunique(),'device':str(device),'qrels_accessed':False},indent=2)+'\n'); print(json.dumps({'stage':'complete','queries':int(frame.query_id.nunique()),'latency_ms_per_query':elapsed*1000/frame.query_id.nunique(),'qrels_accessed':False}))
if __name__=='__main__': main()
