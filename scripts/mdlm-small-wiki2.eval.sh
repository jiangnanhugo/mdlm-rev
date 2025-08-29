checkpoint_path=/home/jiangnan/workspace/mdlm-rev/outputs/wikitext2/2025.08.28/153734/checkpoints-epoch

#### sample eval
for idx in 29999 27999 25999 23999 21999 19999 17999 17999 15999 13999 11999 9999 7999 5999 3999 1999;
do
	for T in 1024 512 256 128 64 32 16 8 4; 
	do
		echo "T = "$T "train-iter = "$idx
		python -m main \
			mode=sample_eval \
			loader.batch_size=1 \
			loader.eval_batch_size=1 \
			data=wikitext2 \
			model=small \
			parameterization=subs \
			backbone=dit \
			model.length=1024 \
			T="$T" \
			sampling.steps=$T \
			eval.checkpoint_path=$checkpoint_path$idx.pt \
			wandb.name=mdlm-wiki2-eval
		done

	done


