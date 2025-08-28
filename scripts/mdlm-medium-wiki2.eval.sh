checkpoint_path=/home/jiangnan/workspace/mdlm-rev/outputs/wikitext2/2025.08.28/153734/checkpint-epoch

#### sample eval
for idx 1999 3999 5999 7999 9999 11999 13999 15999 17999 17999 19999 21999 23999 25999 27999 29999;
do
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

	done


