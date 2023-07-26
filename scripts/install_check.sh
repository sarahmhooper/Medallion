# Run a simple train+eval script to make sure the install was successful; these params are explained in train_and_eval.sh
task=example_task
datapath=../data/${task}
train_segpath=../data/${task}
eval_segpath=../data/${task}
consistency_datapath=../data/${task}
image_size_r=224 
image_size_c=224 
hist_eq=False 
seed=1
augment_k=4
epoch=10
evaluation_freq=5
uncertainty=None    
lr=0.0001
l2=0.00001
min_lr=0.0000001
lr_scheduler=linear
csv_id=None
csv_fn=None
model=UNet
optimizer=adam
batch_size=4
consistency_batch_size=4
scheduler=consistency

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

# Run 
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
      --hist_eq ${hist_eq} \
      --n_epochs ${epoch} \
      --batch_size ${batch_size} \
      --consistency_batch_size ${consistency_batch_size} \
      --valid_batch_size 16 \
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


