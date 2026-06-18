for seed in {0..19}
do
python main.py --graph_type reg --n 1000 --d 3 --k 2 --seed $seed --save 1 --alg ros --wd 1e-4 --lr 1e-2
python main.py --graph_type reg --n 1000 --d 3 --k 2 --seed $seed --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 
python main.py --graph_type reg --n 1000 --d 3 --k 2 --seed $seed --save 1 --alg pignn --lr 1e-4 --patience 1000 --tol 1e-4 --epochs 100000
python main.py --graph_type reg --n 1000 --d 3 --k 2 --seed $seed --save 1 --alg md --epsilon_PMD 1e-8
python main.py --graph_type reg --n 1000 --d 3 --k 2 --seed $seed --save 1 --alg gw
python main.py --graph_type reg --n 1000 --d 3 --k 2 --seed $seed --save 1 --alg genetic
python main.py --graph_type reg --n 1000 --d 3 --k 2 --seed $seed --save 1 --alg bqp
python main.py --graph_type reg --n 1000 --d 3 --k 2 --seed $seed --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20
done