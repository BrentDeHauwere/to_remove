#!/bin/sh

################################3
# Scenario 5
# This SSAD Configuration
for normal_class in 0 1 2 3 4 5 6 7 8 9
do
	for unknown_class in 0 1 2 3 4 5 6 7 8 9
	do
		if [ $normal_class -eq $unknown_class ]; then
      		continue
	    fi
		for dims in 16 32 64 256 512
		do
			CUDA_VISIBLE_DEVICES=3 python baseline_ssad.py mnist ../log/mnist/scenario_5/ssad_hybrid ../data \
				--ratio_known_outlier 0.05 \
				--ratio_pollution 0.1 \
				--kernel rbf \
				--kappa 1.0 \
				--normal_class $normal_class \
				--known_outlier_class $unknown_class \
				--n_known_outlier_classes 1 \
				--seed 0 \
				--feat_dims $dims \
				--case 5 \
				--n_jobs_dataloader 8;

			CUDA_VISIBLE_DEVICES=3 python baseline_ssad.py mnist ../log/mnist/scenario_5/ssad_hybrid ../data \
				--ratio_known_outlier 0.05 \
				--ratio_pollution 0.1 \
				--kernel rbf \
				--kappa 1.0 \
				--hybrid True \
				#--load_ae ../log/cifar-10/scenario_5/deepSAD/model_0_1_${dims}.tar \
				--normal_class $normal_class \
				--known_outlier_class $unknown_class \
				--n_known_outlier_classes 1 \
				--seed 0 \
				--feat_dims $dims \
				--case 5 \
				--n_jobs_dataloader 8;
		done
	done
done
