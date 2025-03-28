import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertConfig
from utils.use import *
import time

# fp32 tensor core
torch.backends.cuda.matmul.allow_tf32 = True
import os
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])

torch.distributed.init_process_group(backend='nccl')
device = torch.device(local_rank)

par = argparse.ArgumentParser(description='training config')
par.add_argument('--batch', type=int, required=True, help='train batch size')
par.add_argument('--iter', type=int, required=True, help='train iter size')
par.add_argument('--config', type=str, required=True, help='train config')
par.add_argument('--logdir', type=str, required=True, help='train log')
args = par.parse_args()

# config
with open(args.config, 'r') as f:
    config = json.load(f)
config = BertConfig(**config)

# data  batch
inputs = torch.rand(args.batch, config.max_position_embeddings, config.hidden_size).to(device)
label = torch.rand_like(inputs).to(device)
dataset = test_dataset(inputs, label)
training_dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

# model
model = Bert_already_emb(config=config).to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# pytorch DP
# model = nn.DataParallel(model)

# DDP
model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)


# train
total_time, be_time, bs_time, fs_time, fe_time = 0, 0, 0, 0, 0
# with torch.profiler.profile(
#         activities=[
#             torch.profiler.ProfilerActivity.CPU,
#             torch.profiler.ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=70, warmup=5, active=3, repeat=0),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=args.logdir),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#         with_flops=True,
#         with_modules=True) as prof:
for step in range(args.iter):
    for i, (input_emb, label) in enumerate(training_dataloader):

        torch.cuda.synchronize()
        fs_time = time.time()

        out_label = model(input_emb)

        # torch.cuda.synchronize()
        # fe_time = time.time()
        
        loss = loss_function(out_label, label)

        # torch.cuda.synchronize()
        # bs_time = time.time()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        be_time = time.time()

        # prof.step()
    if step >= args.iter / 2:
        total_time = total_time + be_time - fs_time
if local_rank == 0:
    print("***")
    print("***")
    print(f'batch_size: {args.batch} bert done',f'valid time={be_time - fs_time}',f'Average time={total_time / (args.iter / 2)}')
    print("***")
    print("***")
