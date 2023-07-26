
task=acdc # Task name (user choice)
datapath=../data/${task} # Directory where image numpys are stored
consistency_datapath=../data/${task} # Directory where image numpys are stored for self-supervised learning, needs to match what was used during model training (typically this is the same as datapath or None if not using unlabeled data)
train_segpath=None # Directory where training seg mask numpys are stored (set to None if evaluating on val/test sets only)
eval_segpath=../data/${task} # Directory where val/test seg mask numpys are stored (typically this is the same as datapath)
train_data_available=False # Indicator for whether to evaluate on train data
val_data_available=True # Indicator for whether to evaluate on validation data
test_data_available=True # Indicator for whether to evaluate on test data
model_path=../logs/acdc_logs/UNet/gt_train_labels/augment_4/with_consistency_loss/without_uncertainty_thresh/csv_acdc_5-keys_all-slice-per-key_seed-1/seed_1/best_model_model_all_val_micro_average.model.pth # Path to model to load for eval 
image_size_r=224 # Num rows in image, should match training param
image_size_c=224 # Num cols in image, should match training param
seed=1 # Random seed       
model=UNet # Model architecture
log_path=../logs/${task}_logs/model_loaded/${model_path:8} # Log path for storing results

# Run evaluation
image_segmentation --task ${task} \
      --datapath ${datapath} \
      --train_segpath ${train_segpath} \
      --consistency_datapath ${consistency_datapath} \
      --eval_segpath ${eval_segpath} \
      --log_path ${log_path} \
      --use_exact_log_path 1 \
      --model ${model} \
      --dataparallel 0 \
      --image_size_r ${image_size_r} \
      --image_size_c ${image_size_c} \
      --valid_batch_size 16 \
      --valid_split val test \
      --seed ${seed} \
      --train 0 \
      --model_path ${model_path} \
      --train_data_available ${train_data_available} \
      --val_data_available ${val_data_available} \
      --test_data_available ${test_data_available} \
      --predict_on_train ${train_data_available} \
      --record_scores True
      
      
