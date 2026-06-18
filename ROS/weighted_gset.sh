
for gset in 70 72 77 81; do
    python main.py --graph_type gset --n 100 --d 3 --k 2 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --gset $gset --weight_mode 2 --lrange 0 --rrange 10

    python main.py --graph_type gset --n 100 --d 3 --k 2 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --gset $gset --weight_mode 2 --lrange 0 --rrange 100



    python main.py --graph_type gset --n 100 --d 3 --k 2 --seed 0 --save 1 --alg cognn --gset $gset --patience 100 --tol 1e-4 --epochs 100000 --lrange 0 --rrange 10 --weight_mode 2 

    python main.py --graph_type gset --n 100 --d 3 --k 2 --seed 0 --save 1 --alg cognn --gset $gset --patience 100 --tol 1e-4 --epochs 100000 --lrange 0 --rrange 100 --weight_mode 2 


    # python main.py --graph_type gset --n 100 --d 3 --k 2 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset $gset --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20 --lrange 0 --rrange 10 --weight_mode 2 

    # python main.py --graph_type gset --n 100 --d 3 --k 2 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset $gset --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20 --lrange 0 --rrange 100 --weight_mode 2 




    # python main.py --graph_type gset --n 100 --d 3 --k 2 --seed 0 --save 1 --alg mcgnn_pretrain --tuning 1 --wd 1e-4 --pretraining_epochs 1 --lr 1e-2 --pretraining_graphnum 500 --gset $gset --pretraining_mode ood --ls 0 --lrange 0 --rrange 10 --weight_mode 2 

    # python main.py --graph_type gset --n 100 --d 3 --k 2 --seed 0 --save 1 --alg mcgnn_pretrain --tuning 1 --wd 1e-4 --pretraining_epochs 1 --lr 1e-2 --pretraining_graphnum 500 --gset $gset --pretraining_mode ood --ls 0 --lrange 0 --rrange 100 --weight_mode 2 
 

done

