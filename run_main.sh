#!/bin/bash

PROJ_NAME="Resic45-ViT_hyp_search"

vit_type="ViT-base"

batch_sizes=(64)
learning_rates=(5e-4)   # 5e-4 3e-4 1e-4
# num_layers=(8)
# heads=(8)
# hiddens=(384)
# mlp_hiddens=(384)
weight_decays=(1e-4 3e-4)
dropouts=(0.2 0.1)
alphas=(0.1 0.2 0.5)
num_opss=(2 4)
magnitudes=(5 10 15 20)

MAX_EPOCH=75
CUDA_GPU=0             # 쿠다 설정 ! 

for bs in ${batch_sizes[@]}
do
    for lr in ${learning_rates[@]}
    do
        for wd in ${weight_decays[@]}
        do
            for drop in ${dropouts[@]}
            do
                for alp in ${alphas[@]}
                do
                    for num_op in ${num_opss[@]}
                    do
                        for mag in ${magnitudes[@]}
                        do

                        EXP_NAME="cvlab11_gpu${CUDA_GPU}_${vit_type}_epochs${MAX_EPOCH}_bs${bs}_lr${lr}_wd${wd}_drop${drop}_alpha${alp}_numop${num_op}_mag${mag}"
                        echo ${EXP_NAME} in progress.. 

                        CUDA_VISIBLE_DEVICES=${CUDA_GPU}  python main.py    --mode train \
                                                                            --data_path /media/dataset_jinlovespho/resisc45 \
                                                                            --dataset resisc45 \
                                                                            --label-smoothing \
                                                                            --mixup \
                                                                            --randaugment \
                                                                            --weight_decay ${wd} \
                                                                            --dropout ${drop} \
                                                                            --alpha ${alp} \
                                                                            --num_ops ${num_op} \
                                                                            --magnitude ${mag} \
                                                                            --epoch ${MAX_EPOCH} \
                                                                            --lr ${lr} \
                                                                            --batch ${bs} \
                                                                            --patch 16 \
                                                                            --model_name ${vit_type} \
                                                                            --project_name ${PROJ_NAME} \
                                                                            --exp_name ${EXP_NAME} \
                                                                            

                        sleep 1m

                        done    
                    done    
                done    
            done    
        done
    done
done 



# for bs in ${batch_sizes[@]}
# do
#     for lr in ${learning_rates[@]}
#     do
#         for head in ${heads[@]}
#         do
#             for num_layer in ${num_layers[@]}
#             do
#                 for hidden in ${hiddens[@]}
#                 do
#                     for mlp_hidden in ${mlp_hiddens[@]}
#                     do
#                         # EXP_NAME="cvlab11_gpu${CUDA_GPU}_bs${bs}_lr${lr}_head${head}_numlayers${num_layer}_hidden${hidden}_mlphidden${mlp_hidden}"
#                         EXP_NAME="cvlab11_gpu${CUDA_GPU}_bs${bs}_lr${lr}_"
#                         echo ${EXP_NAME} in progress.. 

#                         CUDA_VISIBLE_DEVICES=${CUDA_GPU}  python main.py    --mode train \
#                                                                             --data_path /media/dataset_jinlovespho/resisc45 \
#                                                                             --dataset resisc45 \
#                                                                             --epoch ${MAX_EPOCH} \
#                                                                             --lr ${lr} \
#                                                                             --batch ${bs} \
#                                                                             --patch 16 \
#                                                                             --model_name ${vit_type} \
#                                                                             --project_name ${PROJ_NAME} \
#                                                                             --exp_name ${EXP_NAME}
#                         sleep 1m

#                     done
#                 done
#             done
#         done
#     done
# done

