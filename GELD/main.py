import random
from Data.utils import *
from parser_args import *
from main_workers import *
import warnings


def main():

    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset,args.noise_type,str(args.noise_rate),args.arch])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    main_worker_pbar(args)
   

if __name__ == '__main__':
    main()