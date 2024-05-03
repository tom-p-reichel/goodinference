import transformers
import torch
import itertools
from tqdm import tqdm

def embed_batch(model, tokenizer, strings, reduce="last"):
    with torch.cuda.device(model.device):
        args = tokenizer(strings,return_tensors="pt",padding=True)
        out = model(**args)
        # this doesn't work because you can't negative stride in pytorch
        #last = args.attention_mask.shape[1]-1-torch.argmax(args.attention_mask[:,::-1], dim=1).flatten()
        last = args.attention_mask.cumsum(1).argmax(1)
        if reduce == "last":
            # fetch the indexes of the last non-padding tokens
            indices = last+torch.arange(len(last))*out.last_hidden_state.shape[-2]
            return out.last_hidden_state.reshape((-1,out.last_hidden_state.shape[-1]))[indices]
        elif reduce == "mean":
            first = args.attention_mask.argmax(1)
            return torch.vstack([
                out.last_hidden_state[i,f:l+1].mean(axis=0) for i,(f,l) in enumerate(zip(first,last))
            ])
        else:
            raise NotImplementedError(reduce)


@torch.no_grad()
def embed(model,tokenizer,prompts, max_tokens = 1500, progress=False, max_batches = None, reduce="last"):
    lengths = [len(tokenizer(x)["input_ids"]) for x in prompts ]

    order = sorted(range(len(prompts)),key=lambda x: lengths[x])

    """
    if reduce == "last":
        reduce = lambda x : x[:,-1,:]
    elif reduce == "mean":
        reduce = lambda x : x.mean(axis=1)
    else:
        raise NotImplementedError(reduce)
    """

    batches = []
    batch = [prompts[order[0]]]

    for i in tqdm(order[1:]) if progress else order[1:]:
        tokens = lengths[i]*(len(batch)+1)
        if tokens > max_tokens:
            batches.append(embed_batch(model,tokenizer,batch,reduce=reduce).to("cpu"))
            if max_batches is not None and len(batches) >= max_batches:
                batch = []
                break # compute budget exceeded
            batch = [prompts[i]]
        else:
            batch.append(prompts[i])

    if len(batch)>0:
        batches.append(embed_batch(model,tokenizer,batch,reduce=reduce).to("cpu"))
    
    invorder = [None]*len(order)

    for i,io in enumerate(order):
        invorder[io]=i
    
    vecs = torch.vstack(batches)

    if max_batches is None:
        return vecs[invorder]
    else:
        return [i for i in order[:len(vecs)]],vecs




@torch.no_grad()
def embed_naive(model,tokenizer,prompts,progress=False,reduce="last"):
      
    vecs = []
    for x in tqdm(prompts) if progress else prompts:
        out = model(**tokenizer(x,return_tensors="pt").to(model.device))
        if reduce == "last":
            vecs.append(out.last_hidden_state[0,-1,:])
        elif reduce == "mean":
            vecs.append(out.last_hidden_state.mean(dim=1)[0])

    return torch.vstack(vecs).to("cpu")

    

import more_itertools as miter

@torch.no_grad()
def embed_chunk(model,tokenizer,prompts,progress=False,reduce="last",bs=4):
    vecs = []
    for x in tqdm(miter.chunked(prompts,bs)) if progress else miter.chunked(prompts,bs):
        vecs.append(embed_batch(model,tokenizer,x,reduce=reduce))

    return torch.vstack(vecs).to("cpu")




    
def aligns(a,b):
    mat = ((a/torch.linalg.norm(a,dim=1,keepdims=True))@(b/torch.linalg.norm(b,dim=1,keepdims=True)).T)

    print(torch.diag(mat))

    assert((mat.argmax(axis=1) == torch.arange(len(a)).to(a.device)).all())

    assert((torch.diag(mat) >= 0.99).all())



if __name__=="__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    tokenizer.pad_token = tokenizer.eos_token

    model = model.model.to("cuda")

    import random
    
    test_data = []

    words = ["epic","high","low","rock","paper","scissor"]

    for _ in range(100):
        test_data.append(" ".join(random.choices(words,k=random.randint(15,100))))
   
    for reduce in ["mean","last"]:
        
        a = embed(model,tokenizer,test_data,max_tokens=120,progress=True,reduce=reduce)

        a_diff = embed(model,tokenizer,test_data,max_tokens=240,progress=True,reduce=reduce)

        b = embed_naive(model,tokenizer,test_data,progress=True,reduce=reduce)

        c = embed_chunk(model,tokenizer,test_data,progress=True,reduce=reduce)


        aligns(a,b)
        aligns(a,a_diff)
        aligns(a,c)

    
