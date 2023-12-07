import os
import argparse



def get_args_parser():
    parser = argparse.ArgumentParser("Plot",add_help=False)
    parser.add_argument('--checkpoints', default='', type=str)
    parser.add_argument('--wandb', default='', type=str)
    parser.add_argument('--model_type', default='sfa-detr',
                    help='whether to use sfa-detr or detr-only')
    return parser

def main(args): 
    if not args.checkpoints:
        assert "Please provide checkpoints folder path!"
    if not args.wandb:
        assert "Please provide wandb run name!"

    file_paths = os.listdir(args.checkpoints)
    print(file_paths)

    command = "GPUS_PER_NODE=1 DEVICE=cuda python -m train --batch_size 16 --print_feq 10 --eval"
    if len(args.model_type) > 0:
        command = command + " --model_type " + args.model_type
    
    command = command + " --wandb " + args.wandb

    for c in file_paths:
        resume = " --resume " + args.checkpoints + '/' + c
        print(resume)
        os.system(command + resume)

            
    os.system()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Deformable DETR evaluation and plot script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
