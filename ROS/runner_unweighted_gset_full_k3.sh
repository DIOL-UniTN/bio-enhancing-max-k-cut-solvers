for gset in {1..67}; do
    echo $gset 
    python main.py --graph_type gset --k 3 --seed 0 --save 1 --alg ros --wd 1e-4 --lr 1e-2 --gset $gset --weight_mode 1
    python main.py --graph_type gset --k 3 --seed 0 --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --gset $gset --weight_mode 1
    python main.py --graph_type gset --k 3 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --gset $gset --weight_mode 1
    python main.py --graph_type gset --k 3 --seed 0 --save 1 --alg genetic --gset $gset --weight_mode 1
    python main.py --graph_type gset --k 3 --seed 0 --save 1 --alg bqp --gset $gset --weight_mode 1
    python main.py --graph_type gset --k 3 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset $gset --model_dir "./anycsp/models/MAX3CUT" --timeout 180 --num_boost 20 --weight_mode 1
done

