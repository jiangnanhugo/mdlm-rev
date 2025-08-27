#python -u -m main \
#  loader.batch_size=4 \
#  loader.eval_batch_size=4 \
#  model=small \
#  data=wikitext2 \
#  wandb.name=mdlm-wiki2 \
#  parameterization=subs \
#  model.length=32 \
#  eval.compute_generative_perplexity=True \
#  sampling.steps=99



#torchrun --standalone --nproc_per_node=2 -m main \
#  loader.batch_size=4 \
#  loader.eval_batch_size=4 \
#  model=small \
#  data=wikitext2 \
#  wandb.name=mdlm-wiki2 \
#  parameterization=subs \
#  model.length=32 \
#  eval.compute_generative_perplexity=True \
#  sampling.steps=99

checkpoint_path=/home/jiangnan/workspace/mdlm-rev/outputs/wikitext2/2025.08.26/213908/checkpoints-epoch1.pt

checkpoint_path=/home/jiangnan/workspace/mdlm-rev/outputs/wikitext2/2025.08.26/234154/checkpoints-epoch999.pt
T=10
echo "$T"
python -m main \
    mode=ppl_eval \
    loader.batch_size=4 \
    loader.eval_batch_size=4 \
    data=wikitext2 \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=32 \
    T="$T" \
    eval.checkpoint_path=$checkpoint_path \
    wandb.name=mdlm-wiki2-eval
#    +wandb.offline=true


#### sample eval
python -m main \
    mode=sample_eval \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    data=wikitext2 \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=100 \
    T="$T" \
    eval.checkpoint_path=$checkpoint_path \
    wandb.name=mdlm-wiki2-eval