
checkpoint_path=/home/jiangnan/workspace/mdlm-rev/outputs/wikitext2/2025.08.27/011808/checkpoints-epoch9999.pt

#### sample eval
for T in 1 2 3 4 5 6 7 8 9 10; 
do
	echo $T
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
	done
