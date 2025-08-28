checkpoint_path=/home/jiangnan/workspace/mdlm-rev/outputs/wikitext2/2025.08.28/153734/checkpiont-epoch

#### sample eval
for idx in 1999 3999 5999 7999 9999 11999 13999 15999 17999 17999 19999 21999 23999 25999 27999 29999;
do
	for T in 1 10 20 30 40 50 60 70 80 90 100; 
	do
		echo "T = "$T
		python -m main \
			mode=sample_eval \
			loader.batch_size=16 \
			loader.eval_batch_size=16 \
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


