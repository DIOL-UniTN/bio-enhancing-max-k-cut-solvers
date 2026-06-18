python main.py --graph_type COLOR --k 2 --seed 1 --save 1 --alg ros --wd 1e-4 --lr 1e-2 --gset anna
python main.py --graph_type COLOR --k 2 --seed 1 --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --gset anna
python main.py --graph_type COLOR --k 2 --seed 0 --save 1 --alg pignn --gset anna --patience 100 --tol 1e-4 --epochs 100000
python main.py --graph_type COLOR --k 2 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --gset anna
python main.py --graph_type COLOR --k 2 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset anna --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20

python main.py --graph_type COLOR --k 2 --seed 1 --save 1 --alg ros --wd 1e-4 --lr 1e-2 --gset david
python main.py --graph_type COLOR --k 2 --seed 1 --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --gset david
python main.py --graph_type COLOR --k 2 --seed 0 --save 1 --alg pignn --gset david --patience 100 --tol 1e-4 --epochs 100000
python main.py --graph_type COLOR --k 2 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --gset david
python main.py --graph_type COLOR --k 2 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset david --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20

python main.py --graph_type COLOR --k 2 --seed 1 --save 1 --alg ros --wd 1e-4 --lr 1e-2 --gset huck
python main.py --graph_type COLOR --k 2 --seed 1 --save 1 --alg ros_vanilla --wd 1e-4 --lr 1e-2 --gset huck
python main.py --graph_type COLOR --k 2 --seed 0 --save 1 --alg pignn --gset huck --patience 100 --tol 1e-4 --epochs 100000
python main.py --graph_type COLOR --k 2 --seed 0 --save 1 --alg md --epsilon_PMD 1e-8 --gset huck
python main.py --graph_type COLOR --k 2 --seed 0 --save 1 --alg ANYCSP --wd 1e-4 --lr 1e-2 --gset huck --model_dir "./anycsp/models/MAXCUT" --timeout 180 --num_boost 20