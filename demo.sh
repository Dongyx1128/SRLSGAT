# python mains.py train --dataset_name 'Chikusei' --n_blocks 6 --epochs 70 --batch_size 16 --out_feats 256 --n_scale 2 --gpus "0,1"
# python mains.py test --dataset_name 'Chikusei' --n_blocks 6 --out_feats 256 --n_scale 2 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Chikusei' --n_blocks 6 --epochs 70 --batch_size 16 --out_feats 256 --n_scale 4 --gpus "0,1"
# python mains.py test --dataset_name 'Chikusei' --n_blocks 6 --out_feats 256 --n_scale 4 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Chikusei' --n_blocks 6 --epochs 70 --batch_size 16 --out_feats 256 --n_scale 8 --gpus "0,1"
# python mains.py test --dataset_name 'Chikusei' --n_blocks 6 --out_feats 256 --n_scale 8 --cuda 1 --gpus "0,1"

# python mains.py train --dataset_name 'Pavia' --n_blocks 6 --epochs 70 --batch_size 16 --out_feats 256 --n_scale 2 --gpus "0,1"
# python mains.py test --dataset_name 'Pavia' --n_blocks 6 --out_feats 256 --n_scale 2 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Pavia' --n_blocks 6 --epochs 70 --batch_size 16 --out_feats 256 --n_scale 4 --gpus "0,1"
# python mains.py test --dataset_name 'Pavia' --n_blocks 6 --out_feats 256 --n_scale 4 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Pavia' --n_blocks 6 --epochs 70 --batch_size 16 --out_feats 256 --n_scale 8 --gpus "0,1"
# python mains.py test --dataset_name 'Pavia' --n_blocks 6 --out_feats 256 --n_scale 8 --cuda 1 --gpus "0,1"

python mains.py train --dataset_name 'Wdc' --n_blocks 6 --epochs 70 --batch_size 16 --out_feats 256 --n_scale 2 --gpus "0,1"
python mains.py test --dataset_name 'Wdc' --n_blocks 6 --out_feats 256 --n_scale 2 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Wdc' --n_blocks 6 --epochs 70 --batch_size 16 --out_feats 256 --n_scale 4 --gpus "0,1"
# python mains.py test --dataset_name 'Wdc' --n_blocks 6 --out_feats 256 --n_scale 4 --cuda 1 --gpus "0,1"
# python mains.py train --dataset_name 'Wdc' --n_blocks 6 --epochs 70 --batch_size 16 --out_feats 256 --n_scale 8 --gpus "0,1"
# python mains.py test --dataset_name 'Wdc' --n_blocks 6 --out_feats 256 --n_scale 8 --cuda 1 --gpus "0,1"


# Parameter discussion + Ablation study
# python mains.py train --dataset_name 'Pavia' --n_blocks 6 --epochs 70 --out_feats 128 --n_scale 8 --save_dir ./shiyan_model/canshu --gpus "0,1"
# python mains.py train --dataset_name 'Pavia' --n_blocks 6 --epochs 70 --out_feats 128 --n_scale 8 --save_dir ./shiyan_model/xiaorong --gpus "0,1"

# python mains.py test --cuda 1 --gpus "0,1"