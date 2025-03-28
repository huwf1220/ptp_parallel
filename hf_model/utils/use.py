import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, GPT2Model, LlamaModel, MixtralModel
import argparse
import deepspeed
# from transformers.models.gpt2.modeling_gpt2_moe import MoEModel

class my_dataset(Dataset):
    def __init__(self, data, label):
        self.input_ids = data['input_ids']
        self.token_type_ids = data['token_type_ids']
        self.attention_mask = data['attention_mask']
        self.label = label

    def __getitem__(self, item):
        input_ids = self.input_ids[item]
        token_type_ids = self.token_type_ids[item]
        attention_mask = self.attention_mask[item]
        label = self.label[item]
        return input_ids, token_type_ids, attention_mask, label

    def __len__(self):
        return len(self.input_ids)


class test_dataset(Dataset):
    def __init__(self, data, label):
        self.input_ids = data
        self.label = label

    def __getitem__(self, item):
        input_ids = self.input_ids[item]
        label = self.label[item]
        return input_ids, label

    def __len__(self):
        return len(self.input_ids)


class Bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Bert_tune = BertModel(config=config)
        self.tune = nn.Linear(2048, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.Bert_tune(input_ids, attention_mask, token_type_ids)
        pool = out['pooler_output']
        out_label = self.tune(pool)
        return out_label


class Bert_already_emb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Bert = BertModel(config=config)

    def forward(self, input_ids):
        out = self.Bert(inputs_embeds=input_ids)
        hidden_state = out['last_hidden_state']
        return hidden_state


class Llama2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Llama_tune = LlamaModel(config=config)
        self.tune = nn.Linear(config.hidden_size, 10, bias=False)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.Llama_tune(input_ids, attention_mask, token_type_ids)
        hidden_state = out['last_hidden_state']
        output = self.tune(hidden_state)
        return output


class Llama2_already_emb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Llama = LlamaModel(config=config)

    def forward(self, input_ids):
        out = self.Llama(inputs_embeds=input_ids)
        hidden_state = out['last_hidden_state']
        return hidden_state


class gpt2_already_emb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpt2 = GPT2Model(config=config)

    def forward(self, input_ids):
        out = self.gpt2(inputs_embeds=input_ids)
        hidden_state = out['last_hidden_state']
        return hidden_state


class moe_already_emb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moe = MixtralModel(config=config)

    def forward(self, input_ids):
        out = self.moe(inputs_embeds=input_ids)
        hidden_state = out['last_hidden_state']
        return hidden_state



def parse_args():
    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--iter', type=int, required=True, help='train epoch size')
    parser.add_argument('--config', type=str, required=True, help='train config')
    parser.add_argument('--logdir', type=str, required=True, help='train log')
    parser.add_argument('--batch', type=int, required=True, help='train batch size')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()
    return cmd_args
