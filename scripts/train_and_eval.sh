export CUDA_VISIBLE_DEVICES=0

#################### STEP 1: TRAIN INITIAL NETWORK WITH DATA AUGMENTATION + CONSISTENCY LOSS ####################

# Below are parameters you can alter for training
task=acdc # Task name (user defined)
datapath=../data/${task} # Directory where image numpys are stored
train_segpath=../data/${task} # Directory where training seg mask numpys are stored (typically the same as datapath)
eval_segpath=../data/${task} # Directory where val/test seg mask numpys are stored (typically the same as datapath)
consistency_datapath=../data/${task} # Directory where image numpys for consistency loss are stored (often the same as datapath); set to None to not use consistency loss
image_size_r=224 # Num rows in image; code will resize images to this shape
image_size_c=224 # Num cols in image; code will resize images to this shape
seed=1 # Random seed
augment_k=4 # Number of augmentation options to consider for each image; codebase will select hardest of k options to apply
epoch=500 # Number of training epochs
evaluation_freq=5 # Frequency (epochs) to evaluate on val/test set during training
uncertainty=None # Whether to ignore uncertain training seg masks; set to None for step 1
lr=0.0001 # Learning rate
l2=0 # Regularization param
min_lr=0.0000001 # Minimum learning rate, if using LR scheduler; if not using scheduler, this value is ignored
lr_scheduler=None # Learning rate scheduler 
csv_id=acdc_5-keys_all-slice-per-key_seed-${seed} # ID for CSV you're using (should match filename of csv--see next line); set to None to use all available labels
csv_fn=csv_samplers/${csv_id}.csv # Filepath to CSV indicating labeled data; set to None to use all available labels
model=UNet # Model architecture
optimizer=adam # Optimizer for training
batch_size=32 # Batch size for labeled data 
consistency_batch_size=32 # Batch size for unlabeled data, if using
scheduler=consistency # Set to "consistency" if using consistency loss; else set to "augment"

# Below are parameters that are automatically set
if [[ $consistency_datapath == "None" ]]
then
  consistency_setting="without_consistency_loss"
else
  consistency_setting="with_consistency_loss"
fi
if [[ $uncertainty == "None" ]]
then
  uncertainty_setting="without_uncertainty_thresh"
else
  uncertainty_setting="with_uncertainty_thresh_"${uncertainty}
fi
log_path=../logs/${task}_logs/${model}/gt_train_labels/augment_${augment_k}/${consistency_setting}/${uncertainty_setting}/csv_${csv_id}/seed_${seed}

# Run step 1 training
image_segmentation --task ${task} \
      --datapath ${datapath} \
      --train_segpath ${train_segpath} \
      --eval_segpath ${eval_segpath} \
      --consistency_datapath ${consistency_datapath} \
      --log_path ${log_path} \
      --use_exact_log_path 1 \
      --model ${model} \
      --dataparallel 0 \
      --image_size_r ${image_size_r} \
      --image_size_c ${image_size_c} \
      --n_epochs ${epoch} \
      --batch_size ${batch_size} \
      --consistency_batch_size ${consistency_batch_size} \
      --valid_batch_size ${batch_size} \
      --optimizer ${optimizer} \
      --lr ${lr} \
      --l2 ${l2} \
      --min_lr ${min_lr} \
      --lr_scheduler ${lr_scheduler} \
      --valid_split val test \
      --checkpointing 1 \
      --checkpoint_metric model/all/val/micro_average:max \
      --augment_k ${augment_k} \
      --seed ${seed} \
      --train 1 \
      --csv_fn ${csv_fn} \
      --uncertainty ${uncertainty} \
      --evaluation_freq ${evaluation_freq} \
      --predict_on_train True \
      --scheduler ${scheduler} \

##################### RESAVE PSUEDO LABELS ####################
# In this step, we replace the predicted pseudo labels in the train set with the already-available ground truth labels that were used to train step 1

$(python resave_pseudo_labels.py ${log_path}/final_preds/train ${train_segpath}/train ${csv_fn})

##################### STEP 2: TRAIN FINAL MODEL WITH PSEUDO LABELS ####################

# Below are parameters you can alter for training
epoch=250 # Number of epochs for training step 2 network
evaluation_freq=2 # How frequently (in epochs) to evaluate performance
uncertainty=0.15 # Threshold for ignoring uncertain pseudo labels, 0.5+-uncertainty

# Below are parameters that are automatically set, do not alter
train_segpath=${log_path}/final_preds # We'll use the pseudo labels saved in step 1 to train
csv_id=None # We'll use all available pseudo labels
csv_fn=None # We'll use all available pseudo labels
if [[ $uncertainty == "None" ]]
then
  uncertainty_setting="without_uncertainty_thresh"
else
  uncertainty_setting="with_uncertainty_thresh_"${uncertainty}
fi
log_path=../logs/${task}_logs/${model}/pseudo_train_labels/augment_${augment_k}/${consistency_setting}/${uncertainty_setting}/csv_${csv_id}/seed_${seed}

# Run step 2 training
image_segmentation --task ${task} \
      --datapath ${datapath} \
      --train_segpath ${train_segpath} \
      --eval_segpath ${eval_segpath} \
      --consistency_datapath ${consistency_datapath} \
      --log_path ${log_path} \
      --use_exact_log_path 1 \
      --model ${model} \
      --dataparallel 0 \
      --image_size_r ${image_size_r} \
      --image_size_c ${image_size_c} \
      --n_epochs ${epoch} \
      --batch_size ${batch_size} \
      --consistency_batch_size ${consistency_batch_size} \
      --valid_batch_size ${batch_size} \
      --optimizer ${optimizer} \
      --lr ${lr} \
      --l2 ${l2} \
      --min_lr ${min_lr} \
      --lr_scheduler ${lr_scheduler} \
      --valid_split val test \
      --checkpointing 1 \
      --checkpoint_metric model/all/val/micro_average:max \
      --augment_k ${augment_k} \
      --seed ${seed} \
      --train 1 \
      --csv_fn ${csv_fn} \
      --uncertainty ${uncertainty} \
      --evaluation_freq ${evaluation_freq} \
      --predict_on_train False \
      --scheduler ${scheduler} \
      
