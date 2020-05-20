#!/bin/sh

################################3
# Scenario 3
# ocsvm configuration for scneario 3
for normal_class in 0 1 2 3 4 5 6 7 8 9
do
	for unknown_class in 0 1 2 3 4 5 6 7 8 9
	do
		if [ $normal_class -eq $unknown_class ]; then
      		continue
    	fi
		for kappa in 0 1 2 3 5 # QUESTION: NOT USED?
		do
			# OC-SVM
			CUDA_VISIBLE_DEVICES=3 python baseline_ocsvm.py mnist ../log/mnist/scenario_1/ocsvm ../data \
				--ratio_known_outlier $gamma_l \ # QUESTION: NOT DEFINED 0.05?
				--ratio_pollution 0.1 \
				--kernel rbf \
				--normal_class $normal_class \
				--known_outlier_class $unknown_class \
				--n_known_outlier_classes 1 \ # this kappa?
				--seed 0 \
                --case 1 \ # CASE 3?
				--n_jobs_dataloader 8;

	# 		# OC-SVM Hybrid
			CUDA_VISIBLE_DEVICES=3 python baseline_ocsvm.py mnist ../log/mnist/scenario_1/ocsvmHybrid ../data \
			--ratio_known_outlier $gamma_l \
			--ratio_pollution 0.1 \
			--kernel rbf \
			--normal_class $normal_class \
			--known_outlier_class $unknown_class \
			--n_known_outlier_classes 1 \
            --seed 0 \
			--hybrid True \
			#--load_ae ../log/cifar-10/scenario_1/deepSAD/model_${normal_class}_${unknown_class}_0.tar \
            --case 1 \ # CASE 3?
			--n_jobs_dataloader 8
		done
	done
done
