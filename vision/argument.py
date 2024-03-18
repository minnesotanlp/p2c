import argparse

def parser():

    parser = argparse.ArgumentParser(description='PyTorch P2C Learning in Vision')
    parser.add_argument('--train_type', default='vanilla', type=str, help='standard')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate)
    parser.add_argument('--dataset', default='sun', type=str)
    parser.add_argument('--data_root', default='./images', type=str, help='PATH TO SUN dataset')
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--model', default="resnet18", type=str)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--data_type', default=0, type=int, help='SUN data index')

    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epoch', default=100, type=int, 
                        help='total epochs to run')
    parser.add_argument('--decay', default=1e-2, type=float, help='weight decay')

    parser.add_argument("--lambda_cls", help='weight for classification loss',
                        default=1.0, type=float)
    parser.add_argument("--lambda_pref", help='weight for preference loss',
                        default=1.0, type=float)
    parser.add_argument("--lambda_div", help='weight for diversity regularization',
                        default=1.0, type=float)
    parser.add_argument("--lambda_del", help='weight for consistency regularization',
                        default=1.0, type=float)
    
    parser.add_argument("--consistency", help='independent ensemble loss',
                        action='store_true')
    parser.add_argument("--static", help='independent ensemble loss',
                        action='store_true')
    parser.add_argument("--pair_loss", help='independent ensemble loss',
                        action='store_true')
    
    args = parser.parse_args()

    return args

def print_args(args):
    for k, v in vars(args).items():
        print('{:<16} : {}'.format(k, v))

