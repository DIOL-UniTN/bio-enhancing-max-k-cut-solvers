for seed in {0..9}; do
    for gset in 70 72 77 81; do
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg ros --wd 1e-4 --lr 1e-2 --gset $gset --weight_mode 2 --lrange 0.9 --rrange 1.1
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --gset $gset --weight_mode 2 --lrange 0.9 --rrange 1.1
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg md --epsilon_PMD 1e-8 --gset $gset --weight_mode 2 --lrange 0.9 --rrange 1.1
        python main.py --graph_type gset --k 3  -seed $seed --save 1 --alg genetic --gset $gset --weight_mode 2 --lrange 0.9 --rrange 1.1
        python main.py --graph_type gset --k 3  -seed $seed --save 1 --alg bqp --gset $gset --weight_mode 2 --lrange 0.9 --rrange 1.1
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset $gset --model_dir "./anycsp/models/MAX3CUT" --timeout 180 --num_boost 20 --weight_mode 2 --lrange 0.9 --rrange 1.1
    done
done

for seed in {0..9}; do
    for gset in 70 72 77 81; do
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg ros --wd 1e-4 --lr 1e-2 --gset $gset --weight_mode 2 --lrange 0 --rrange 10
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --gset $gset --weight_mode 2 --lrange 0 --rrange 10
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg md --epsilon_PMD 1e-8 --gset $gset --weight_mode 2 --lrange 0 --rrange 10
        python main.py --graph_type gset --k 3  -seed $seed --save 1 --alg genetic --gset $gset --weight_mode 2 --lrange 0 --rrange 10
        python main.py --graph_type gset --k 3  -seed $seed --save 1 --alg bqp --gset $gset --weight_mode 2 --lrange 0 --rrange 10
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset $gset --model_dir "./anycsp/models/MAX3CUT" --timeout 180 --num_boost 20 --weight_mode 2 --lrange 0 --rrange 10
    done
done

for seed in {1..9}; do
    for gset in 70 72 77 81; do
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg ros --wd 1e-4 --lr 1e-2 --gset $gset --weight_mode 2 --lrange 0 --rrange 100
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --gset $gset --weight_mode 2 --lrange 0 --rrange 100
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg md --epsilon_PMD 1e-8 --gset $gset --weight_mode 2 --lrange 0 --rrange 100
        python main.py --graph_type gset --k 3  -seed $seed --save 1 --alg genetic --gset $gset --weight_mode 2 --lrange 0 --rrange 100
        python main.py --graph_type gset --k 3  -seed $seed --save 1 --alg bqp --gset $gset --weight_mode 2 --lrange 0 --rrange 100
        python main.py --graph_type gset --k 3 --seed $seed --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset $gset --model_dir "./anycsp/models/MAX3CUT" --timeout 180 --num_boost 20 --weight_mode 2 --lrange 0 --rrange 100
    done
done