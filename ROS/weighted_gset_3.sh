
for gset in 70 72 77 81; do
    python main.py --graph_type gset --n 100 --d 3 --k 3 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --gset $gset --weight_mode 2 --lrange 0.5 --rrange 1.5

    python main.py --graph_type gset --n 100 --d 3 --k 3 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --gset $gset --weight_mode 2 --lrange 90 --rrange 110

    python main.py --graph_type gset --n 100 --d 3 --k 3 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --gset $gset --weight_mode 2 --lrange 50 --rrange 150


    python main.py --graph_type gset --n 100 --d 3 --k 3 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset $gset --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20 --lrange 0.5 --rrange 1.5 --weight_mode 2 

    python main.py --graph_type gset --n 100 --d 3 --k 3 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset $gset --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20 --lrange 90 --rrange 110 --weight_mode 2 

    python main.py --graph_type gset --n 100 --d 3 --k 3 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset $gset --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20 --lrange 50 --rrange 150 --weight_mode 2 



    python main.py --graph_type gset --n 100 --d 3 --k 3 --seed 0 --save 1 --alg mcgnn_pretrain --tuning 1 --wd 1e-4 --pretraining_epochs 1 --lr 1e-2 --pretraining_graphnum 500 --gset $gset --pretraining_mode ood --ls 0 --lrange 0.5 --rrange 1.5 --weight_mode 2 

    python main.py --graph_type gset --n 100 --d 3 --k 3 --seed 0 --save 1 --alg mcgnn_pretrain --tuning 1 --wd 1e-4 --pretraining_epochs 1 --lr 1e-2 --pretraining_graphnum 500 --gset $gset --pretraining_mode ood --ls 0 --lrange 90 --rrange 110 --weight_mode 2 

    python main.py --graph_type gset --n 100 --d 3 --k 3 --seed 0 --save 1 --alg mcgnn_pretrain --tuning 1 --wd 1e-4 --pretraining_epochs 1 --lr 1e-2 --pretraining_graphnum 500 --gset $gset --pretraining_mode ood --ls 0 --lrange 50 --rrange 150 --weight_mode 2 

done

