
import unittest
import os
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import alpa
from alpa import parallelize, ShardParallel, LocalPhysicalDeviceMesh, AutoShardingOption
from alpa.model.bert_model import (BertConfig, FlaxBertLayerCollection,
                                   FlaxBertForMaskedLMModule)
from alpa.device_mesh import VirtualPhysicalMesh
import argparse

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# head = num_heads,
# NB = num_micro_batches, PM = parallel_mode
# 3D config = 3D parallel config (Data, Operator, Pipeline)
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,
GPTModelConfig = namedtuple(
    "GPTModelConfig",
    ["seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size"])

gpt_specs = {
    #                      S，   H,   L,  head,   V,
    "A": GPTModelConfig(1024, 768, 12, 12, 51200),  # 125M
    "B": GPTModelConfig(1024, 1024, 24, 16, 51200),  # 350M
    "C": GPTModelConfig(1024, 1536, 24, 16, 51200),  # 760M
    "D": GPTModelConfig(1024, 2048, 24, 32, 51200),  # 1.3B
    "E": GPTModelConfig(1024, 2560, 32, 32, 51200),  # 2.6B
    "F": GPTModelConfig(1024, 4096, 32, 32, 51200),  # 6.7B
    "G": GPTModelConfig(1024, 5120, 48, 40, 51200),  # 15B
    "H": GPTModelConfig(1024, 8192, 48, 64, 51200),  # 39B
    "I": GPTModelConfig(1024, 10240, 60, 80, 51200),  # 76B
}


class AutoShardingAttentionTest(unittest.TestCase):
    def setUp(self, config_str="A", batch_size=32, num_layers=1, device_num=4, data_type=jnp.float32, debug_dir=""):
        if len(jax.local_devices()) >= device_num:
            self.physical_mesh = LocalPhysicalDeviceMesh(
                jax.local_devices()[:device_num])
        else:  # only for test compile and debug.
            self.physical_mesh = VirtualPhysicalMesh(
                ["host_0"], [{"NodeManagerAddress": "1.1.1.1"}], device_num)
        mesh_shape_1d = [1, device_num]
        self.device_mesh_1d = self.get_device_mesh(
            mesh_shape_1d, [1, 1], [1, 1])
        mesh_shape_2d = [2, device_num // 2]
        self.device_mesh_2d = self.get_device_mesh(
            mesh_shape_2d, [1, 1], [1, 1])
        self.model_config = gpt_specs[config_str]
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.data_type = data_type
        self.as_option = AutoShardingOption()
        self.as_option.force_batch_dim_to_mesh_dim = -1
        if (debug_dir != ""):
            self.as_option.print_strategy = True
            self.as_option.debug_dir = os.getcwd()+'/'+debug_dir


    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_bert(self, batch_size, seq_len, num_layers, hidden_size,
                        num_heads, deterministic, use_remat, device_mesh):

        @parallelize(method=ShardParallel(devices=device_mesh,
                                          auto_sharding_option=self.as_option))
        def train_step(state, batch, deterministic):

            def loss_func(params):
                rngs = {"dropout": batch["rng"]}
                out = state.apply_fn(params,
                                     batch["hidden_states"],
                                     batch["attention_mask"],
                                     deterministic,
                                     rngs=rngs)[0]
                return jnp.mean((out - batch["label"])**2)

            grads = alpa.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        hidden_states = jnp.ones((batch_size, seq_len, hidden_size),
                                 dtype=self.data_type)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        label = jnp.ones((batch_size, seq_len, hidden_size), dtype=self.data_type)

        model = FlaxBertLayerCollection(
            BertConfig(num_hidden_layers=num_layers,
                       hidden_size=hidden_size,
                       intermediate_size=hidden_size * 4,
                       num_attention_heads=num_heads,
                       gradient_checkpointing=use_remat, 
                       ),
                       dtype=self.data_type)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, hidden_states, attention_mask)
        tx = optax.adam(1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        state = train_step(
            state, {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "label": label,
                "rng": rngkey
            }, deterministic)

        executable = train_step.get_last_executable()
        return (state, executable.get_hlo_text(),
                executable.auto_sharding_objective)

    def test_bert(self):
        state, hlo_ir, objective = self.run_bert(self.batch_size, self.model_config.seq_len,
                                                self.num_layers, self.model_config.hidden_size,
                                                self.model_config.num_heads,
                                                False, False, self.device_mesh_1d)


def bert_alpa_default_perf_test(config, batch_size, num_layer, device_num, data_type, debug_dir=""):
    bert_test = AutoShardingAttentionTest()
    bert_test.setUp(config, batch_size, num_layer, device_num, data_type, debug_dir)
    bert_test.test_bert()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="C", help="BERT model configuration (A, B, C, ... I)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_layer", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--device_num", type=int, default=4, help="Number of devices to use")
    parser.add_argument("--data_type", type=str, default="float32", choices=["float32", "float16"], help="Data type")
    parser.add_argument("--debug_dir", type=str, default="tmp", help="Directory for debugging output")
    parser.add_argument("--dump_comm_volume", type=str, choices=["true", "false"], default="false", help="Enable/disable DUMP_COMM_VOLUME")
    args = parser.parse_args()

    os.environ["DUMP_COMM_VOLUME"] = args.dump_comm_volume

    data_type = jnp.float32 if args.data_type == "float32" else jnp.float16
    print(f"============ Alpa BERT test: config:{args.config} batchsize:{args.batch_size} layers:{args.num_layer} devices:{args.device_num} ============")
    bert_alpa_default_perf_test(args.config, args.batch_size, args.num_layer, args.device_num, data_type, args.debug_dir)

