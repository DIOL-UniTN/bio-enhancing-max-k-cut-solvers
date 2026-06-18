python main.py --graph_type bitcoin --k 2 --seed 0 --save 1 --alg ros --wd 1e-4 --lr 1e-2 --weight_mode 2
python main.py --graph_type bitcoin --k 2 --seed 0 --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --weight_mode 2
python main.py --graph_type bitcoin --k 2 --seed 0 --save 1 --alg pignn --patience 100 --tol 1e-4 --epochs 100000 --weight_mode 2
python main.py --graph_type bitcoin --k 2 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --weight_mode 2
python main.py --graph_type bitcoin --k 2 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20 --weight_mode 2

python main.py --graph_type bitcoin --k 3 --seed 0 --save 1 --alg ros --wd 1e-4 --lr 1e-2 --weight_mode 2
python main.py --graph_type bitcoin --k 3 --seed 0 --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --weight_mode 2
python main.py --graph_type bitcoin --k 3 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --weight_mode 2
python main.py --graph_type bitcoin --k 3 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --model_dir "./anycsp/models/MAX3CUT" --timeout 180 --num_boost 20 --weight_mode 2


python main.py --graph_type bitcoin --k 10 --seed 0 --save 1 --alg ros --wd 1e-4 --lr 1e-2 --weight_mode 2
python main.py --graph_type bitcoin --k 10 --seed 0 --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --weight_mode 2
python main.py --graph_type bitcoin --k 10 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --weight_mode 2
python main.py --graph_type bitcoin --k 10 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --model_dir "./anycsp/models/MAX10CUT" --timeout 180 --num_boost 20 --weight_mode 2