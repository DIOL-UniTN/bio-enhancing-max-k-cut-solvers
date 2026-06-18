import torch
from utils import set_random_seed
from add_parser import add_parse
from ros.utils import get_gnn, pretraining


if __name__ == "__main__":
    args = add_parse()
    set_random_seed(args.seed)
    args.TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.TORCH_DTYPE = torch.float32
    net, optimizer, scheduler = get_gnn(args, args.TORCH_DEVICE, args.TORCH_DTYPE)
    net = pretraining(args, net, optimizer, scheduler, args.pretraining_epochs, args.pretraining_graphnum)
        