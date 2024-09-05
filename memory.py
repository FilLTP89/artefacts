import torch
import gc
import inspect
import functools
import warnings
warnings.filterwarnings("ignore", message="The default value of the antialias parameter", category=UserWarning)

def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False



def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from accelerate.utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                else:
                    raise

    return decorator


if __name__ == "__main__":
    from model.torch.Attention_MEDGAN import VGG19 
    from data_file.processing_newdata import ClassificationDataset, Datav2Module
    import torch.nn as nn   
    model = VGG19(
        n_class=2,
        classifier_training=True,
    )    
    module = Datav2Module(
        dataset_type=ClassificationDataset,
        train_bs=128
    )
    module.setup()
    data_loader = module.train_dataloader()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    @find_executable_batch_size(starting_batch_size=128)
    def train_step(batch_size, model, optimizer, data_loader):
        model.train()
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs[:batch_size].to("cuda")  # Limit batch to the current batch_size
            targets = targets[:batch_size].to("cuda")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            print(f"Training with batch size: {batch_size}")
            return loss.item()  # Return loss for this batch
    
    loss = train_step(model, optimizer, data_loader)