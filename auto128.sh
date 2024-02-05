#!/bin/bash

# 파라미터 조합
model_names=("ViT-tiny" "ViT-small")
batch_sizes=(128)
learning_rates=(5e-4 3e-4 1e-4)
weight_decays=(1e-4 3e-4)
dropouts=(0.2 0.1)
alphas=(0.1 0.2 0.5)
num_opss=(2 4)
magnitudes=(5 10 15 20)

# 동시에 실행할 최대 실험 수
MAX_JOBS=1

# 각 조합에 대해 실험 실행
for model_name in "${model_names[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for weight_decay in "${weight_decays[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    for alpha in "${alphas[@]}"; do
                        for num_ops in "${num_opss[@]}"; do
                            for magnitude in "${magnitudes[@]}"; do
                                # 백그라운드 작업 수 확인 및 대기
                                while [ $(jobs | wc -l) -ge $MAX_JOBS ]; do
                                    sleep 1
                                done

                                # 실험 실행
                                python main.py --label-smoothing --mixup --randaugment --model_name $model_name --batch $batch_size --lr $lr --weight_decay $weight_decay --dropout $dropout --alpha $alpha --num_ops $num_ops --magnitude $magnitude &
                            done
                        done
                    done
                done
            done
        done
    done
done

# 모든 백그라운드 작업이 완료될 때까지 대기
wait
echo "All experiments completed."