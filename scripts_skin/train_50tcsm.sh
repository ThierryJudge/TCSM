cd ..
python train_tcsm_mean.py --gpu 3 --out exp/skin/skin50_tcsm  --scale  --lr 1e-4 --n-labeled 50 --consistency 1.0 --consistency_rampup 600 --epochs 800   --batch-size 22  \
--num-class 2  --val-iteration 10
