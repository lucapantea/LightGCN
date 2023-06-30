import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def create_optimizer(model, args):
    """
    Create an optimizer for model parameters.

    Args:
        model (torch.nn.Module): The model for which to create the optimizer.
        args (argparse.Namespace): Optimizer information.

    Returns:
        torch.optim.Optimizer: The created optimizer.

    Raises:
        ValueError: If the specified optimizer is unsupported.
    """
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    return optimizer


def create_scheduler(optimizer, args):
    """
    Create a learning rate scheduler for the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for the scheduler.
        args (argparse.Namespace): Scheduler information.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler.

    Raises:
        ValueError: If the specified scheduler is unsupported.
    """
    if args.scheduler == "step_lr":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_params.get("step_size", 10),
            gamma=args.scheduler_params.get("gamma", 0.1))
    elif args.scheduler == "reduce_lr_on_plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            patience=args.scheduler_params.get("patience", 5))
    elif args.scheduler == "cosine_annealing":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.scheduler_params.get("T_max", 10))
    elif args.scheduler == "one_cycle_lr":
        max_lr = args.scheduler_params.get("max_lr", 0.01)
        epochs = args.scheduler_params.get("epochs", 100)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr, epochs)
    elif args.scheduler == "cosine_annealing_warm_restarts":
        T_0 = args.scheduler_params.get("T_0", 10)
        T_mult = args.scheduler_params.get("T_mult", 2)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0, T_mult)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    return scheduler
