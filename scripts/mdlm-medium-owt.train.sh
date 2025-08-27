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



torchrun --standalone --nproc_per_node=2 -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=medium \
  data=openwebtext-split \
  wandb.name=mdlm-owt \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000