#!/usr/bin/env python

import os
import json
import pprint as pp
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from options import get_options
from train import train_epoch, validate
from nets.reinforce_baselines import RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
# from nets.pointer_network import PointerNetwork
from utils import torch_load_cpu, load_problem, get_inner_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def run(opt):

    # Pretty print the run args
    pp.pprint(vars(opt))

    ##

    # Set the random seed
    # torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opt.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opt.log_dir, "{}_{}".format(opt.problem, opt.n_users), opt.run_name))

    os.makedirs(opt.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opt.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opt), f, indent=True)

    # Set the device
    opt.device = torch.device("cuda:0" if opt.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opt.problem)

    # Load model parameters from load_path
    load_data = {}
    assert opt.load_path is None or opt.resume is None, "Only one of load path and resume can be given"
    load_path = opt.load_path if opt.load_path is not None else opt.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        # 'pointer': PointerNetwork,
        'attention': AttentionModel
    }.get(opt.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opt.embedding_dim,
        opt.hidden_dim,
        problem,
        n_encode_layers=opt.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opt.normalization,
        tanh_clipping=opt.tanh_clipping,
        checkpoint_encoder=opt.checkpoint_encoder,
        shrink_size=opt.shrink_size,
        dy=False
    ).to(opt.device)

    if opt.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load q
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    baseline = RolloutBaseline(model, problem, opt)

    if opt.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opt.bl_warmup_epochs, warmup_exp_beta=opt.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opt.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opt.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opt.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opt.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(n_users=opt.n_users, n_facilities=opt.n_facilities, num_samples=opt.val_size,
                                       filename='./data/MCLP/MCLP_1500_25/MCLP_1500_25_valid_Normalization.pkl', p=opt.p,
                                       r=opt.r, distribution=opt.data_distribution)

    if opt.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opt.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opt.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opt.epoch_start = epoch_resume + 1

    if opt.eval_only:
        validate(model, val_dataset, opt)
    else:
        for epoch in range(opt.epoch_start, opt.epoch_start + opt.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opt
            )


if __name__ == "__main__":
    run(get_options())
