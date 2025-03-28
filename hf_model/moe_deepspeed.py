import json
import torch
import torch.optim as optim
from transformers import MixtralConfig
from utils.use import *
import time
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True

# cmd_args
cmd_args = parse_args()

# config
with open(cmd_args.config, 'r') as f:
    config = json.load(f)
config = MixtralConfig(**config)

with open(cmd_args.deepspeed_config, 'r') as f:
    zero_config = json.load(f)
zero_config = MixtralConfig(**zero_config)

import os
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
device = torch.device(local_rank)

# data  batch
# data = torch.rand(zero_config.train_batch_size, config.max_position_embeddings, config.hidden_size)
# label = torch.ones_like(data)
# dataset = test_dataset(data, label)

inputs = torch.rand(cmd_args.batch, config.max_position_embeddings, config.hidden_size).to(device)
label = torch.rand_like(inputs).to(device)
dataset = test_dataset(inputs, label)
training_dataloader = DataLoader(dataset, batch_size=cmd_args.batch, shuffle=True)

# model
model = moe_already_emb(config=config)

optimizer = optim.Adam(model.parameters(), lr=0.1)
engine, optimizer, _, _ = deepspeed.initialize(
    args=cmd_args,
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer
)
loss_function = nn.CrossEntropyLoss().to(engine.device)
# train
total_time, fs_time, fe_time, be_time, bs_time = 0, 0, 0, 0, 0
# with torch.profiler.profile(
#         activities=[
#             torch.profiler.ProfilerActivity.CPU,
#             torch.profiler.ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=0, warmup=15, active=3, repeat=0),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=cmd_args.logdir),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#         with_flops=True,
#         with_modules=True) as prof:
for epoch in range(cmd_args.iter):
    for i, (x, y) in enumerate(training_dataloader):
        torch.cuda.synchronize()
        fs_time = time.time()
        out_label = engine(x)

        # torch.cuda.synchronize()
        # fe_time = time.time()

        loss = loss_function(out_label, y)

        # torch.cuda.synchronize()
        # bs_time = time.time()

        engine.backward(loss)

        engine.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        be_time = time.time()

        # prof.step()

    if epoch >= cmd_args.iter / 2:
        total_time = total_time + be_time - fs_time

if local_rank == 0:
    print("***")
    print("***")
    print(f'batch_size: {cmd_args.batch} moe done',f'valid time={be_time - fs_time}',f'Average time={total_time / (cmd_args.iter / 2)}')
    print("***")
    print("***")
