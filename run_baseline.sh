#!/usr/bin/env bash

for i in {1..10}
do
# use gpu id 2 for running the experiment
ngpus=2
# run for 30 epochs
epochs=30
exp_name='hcc'
# three fold cross validation, baseline_num is the particular fold's experiment
exp_num='baseline_1'
# use resnet-18 based encoder
bool_use_resnet=1
#bool_use_resnet=0

pre_train_sup=0
method=''
sup_model=''
chkpt_path=''
sup_model_path=''

# location of the data
data_dir='../pred_116/data/data_folds_2d_three_res'
img_dir=${data_dir}

for exp_num in 'baseline_1' 'baseline_2' 'baseline_3'
do
# ours is a 3 class problem
num_class=3
# this code trains the models
python train.py  --random-seed -1 --epochs ${epochs}  --exp ${exp_name} --exp-num ${exp_num} --model Classifier --num-class ${num_class} --batch-size 4 \
  --save-dir ./experiments/${exp_name}/${exp_num} --gpus ${ngpus} --use_resnet ${bool_use_resnet} --data-dir ${data_dir} \
  --pre_train_sup ${pre_train_sup}

# this code tests the models saved from the previous step
python test.py --exp ${exp_name} --exp-num ${exp_num} --model Classifier --num-class ${num_class} --batch-size 4  \
  --model-path ./experiments/${exp_name}/${exp_num}/checkpoint_best.pth.tar  \
  --save-dir ./experiments/${exp_name}/${exp_num}/best --gpus ${ngpus} --use_resnet ${bool_use_resnet} --img-dir ${img_dir}
done
#done

# collect the results
# these are  mainly for formatting the result nicely
python parse_results.py
python create_excel.py --chkpt_path ${chkpt_path}
exps='./experiments'
all_exps='../exp_all'
excel='output.xlsx'
mv ${exps} ${exps}'_'$i
mv $excel $i'_'$excel
done
python combine_multiple_excels.py
