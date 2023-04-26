import numpy as np
import torch
import argparse
import datetime
import wandb
from utils import utils
from algorithms import run_sac, run_td3


if __name__ == "__main__":
    main_start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    utils.add_arguments(parser)
    args = parser.parse_args()
    utils.print_all_args(args)
    utils.make_folders()
    file_name = utils.set_file_name(args)
    wandb.init(project="ANF", config=vars(args), name=file_name, entity="VScAIL", mode=args.wandb_mode)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if 'TD3' in args.policy:
        run_td3.run(args, file_name, device)
    elif 'SAC' in args.policy:
        run_sac.run(args, file_name, device)
    else:
        raise ValueError("Invalid policy name")

    total_runtime = datetime.datetime.now() - main_start_time
    print(f"\nTotal running time {total_runtime}\n\n")
    wandb.log({"total_runtime_hours": round(total_runtime.total_seconds() / 3600, 2)})
