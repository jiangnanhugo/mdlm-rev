checkpoint_path=XXXXXXXXXXXXXXXXXXXXXXXXXX

#### sample eval
for T in 1 10 20 30 40 50 60 70 80 90 100; 
do
	echo "T = "$T
	python -m main \
		mode=sample_eval \
		loader.batch_size=16 \
    loader.eval_batch_size=16 \
		data=openwebtext-split \
		model=small \
		parameterization=subs \
		backbone=dit \
		model.length=1024 \
		T="$T" \
		eval.checkpoint_path=$checkpoint_path \
		wandb.name=mdlm-owt-eval
done




