CUDA_VISIBLE_DEVICES=0  python main.py      --mode train \
                                            --data_path /mnt/ssd2/dataset/resisc45 \
                                            --dataset resisc45 \
                                            --epoch 75 \
                                            --lr 1e-4 \
                                            --batch 64 \
                                            --patch 16 \
                                            --model_name ViT-small \
                                            --project_name pho_Resic45-ViT



