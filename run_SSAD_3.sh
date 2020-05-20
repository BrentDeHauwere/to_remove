#!/bin/sh

################################3


for normal_class in 0 1 2 3 4 5 6 7 8 9
do
	for seed in 0 1 2 3 4 5 6 7 8 9
	do
		for kappa in 0 1 2 3 5
		do
			CUDA_VISIBLE_DEVICES=3 python baseline_ssad.py mnist ../log/mnist/scenario_3/ssad_raw ../data \
				--ratio_known_outlier 0.05 \
				--ratio_pollution 0.1 \
				--kernel rbf \
				--kappa 1.0 \
				--normal_class $normal_class \
				--known_outlier_class 0 \
				--n_known_outlier_classes $kappa \
				--seed $seed \
				--case 3 \
				--n_jobs_dataloader 8;

			CUDA_VISIBLE_DEVICES=3 python baseline_ssad.py mnist ../log/mnist/scenario_3/ssad_hybrid ../data \
				--ratio_known_outlier 0.05 \
				--ratio_pollution 0.1 \
				--kernel rbf \
				--kappa 1.0 \
				--hybrid True \
				# --load_ae ../log/cifar-10/scenario_2/deepSAD/model_0_1_005.tar \
				--normal_class $normal_class \
				--known_outlier_class 0 \
				--n_known_outlier_classes $kappa \
				--seed $seed \
				--case 3 \
				--n_jobs_dataloader 8;
		done
	done
done

