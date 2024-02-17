#!/bin/bash

PROJ_NAME="20240214_Resic45-ViT_hyp_search"

vit_type="ViT-small"     # ViT-tiny, ViT-small, ViT-base, ViT-large, ViT-huge 중에서 선택 

batch_sizes=(128)
learning_rates=(1e-6)   # 1e-5 5e-5 1e-4 3e-4 5e-4
weight_decays=(1e-3 1e-4 1e-5 1e-6)
dropouts=(0.3 0.5 0.7)
alphas=(0.3 0.5 0.7)
warm_ups=(5 10)


MAX_EPOCH=80
CUDA_GPU=3            # 쿠다 설정 ! 

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
                    for wu in ${warm_ups[@]}
                    do

                    EXP_NAME="cvlab11_gpu${CUDA_GPU}_${vit_type}_epochs${MAX_EPOCH}_bs${bs}_lr${lr}_wd${wd}_drop${drop}_alpha${alp}_warmup${wu}"
                    echo ${EXP_NAME} in progress.. 

                    CUDA_VISIBLE_DEVICES=${CUDA_GPU}  python main.py    --mode train \
                                                                        --data_path /media/dataset_jinlovespho/resisc45 \
                                                                        --dataset resisc45 \
                                                                        --label-smoothing \
                                                                        --mixup \
                                                                        --weight_decay ${wd} \
                                                                        --dropout ${drop} \
                                                                        --alpha ${alp} \
                                                                        --epoch ${MAX_EPOCH} \
                                                                        --lr ${lr} \
                                                                        --batch ${bs} \
                                                                        --model_name ${vit_type} \
                                                                        --project_name ${PROJ_NAME} \
                                                                        --exp_name ${EXP_NAME} \
                                                                        
                    sleep 30s

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

