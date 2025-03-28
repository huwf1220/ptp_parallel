"""Test auto sharding on transformer layers and bert models."""

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
from alpa.shard_parallel.pb_config import PBConfig, default_pb_config
from alpa.pipeline_parallel.layer_construction import manual_layer_construction
from dataclasses import replace
import argparse

GPTModelConfig = namedtuple(
    "GPTModelConfig",
    ["seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size"])

gpt_specs = {
    #                      S，   H,   L,  head,   V,
    "B": GPTModelConfig(1024, 1024, 24, 16, 51200),  # 350M
    "C": GPTModelConfig(1024, 1536, 24, 16, 51200),  # 760M
    "D": GPTModelConfig(1024, 2048, 24, 32, 51200),  # 1.3B
    "E": GPTModelConfig(1024, 2560, 32, 32, 51200),  # 2.6B
}


class AutoShardingAttentionTest(unittest.TestCase):

    def setUp(self,
              config_str="A",
              batch_size=32,
              num_layers=1,
              device_num=4,
              data_type=jnp.float32,
              debug_dir=""):
        if len(jax.local_devices()) >= device_num:
            self.physical_mesh = LocalPhysicalDeviceMesh(
                jax.local_devices()[:device_num])
        else:  # only for test compile and debug.
            self.physical_mesh = VirtualPhysicalMesh(["host_0"], [{
                "NodeManagerAddress": "1.1.1.1"
            }], device_num)
        mesh_shape_1d = [1, device_num]
        self.device_mesh_1d = self.get_device_mesh(mesh_shape_1d, [1, 1],
                                                   [1, 1])
        mesh_shape_2d = [2, device_num // 2]
        self.device_mesh_2d = self.get_device_mesh(mesh_shape_2d, [1, 1],
                                                   [1, 1])
        self.data_type = data_type
        self.model_config = gpt_specs[config_str]
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.as_option = AutoShardingOption()
        self.as_option.mode = "default"
        self.as_option.num_layers = num_layers
        self.as_option.flop_count = (
            72.0 * batch_size * self.model_config.seq_len * self.num_layers *
            self.model_config.hidden_size * self.model_config.hidden_size *
            (1.0 + self.model_config.seq_len /
             (6.0 * self.model_config.hidden_size)) / 1e12)
        self.as_option.pb_config = default_pb_config
        self.as_option.force_batch_dim_to_mesh_dim = -1
        if (debug_dir != ""):
            self.as_option.print_strategy = True
            self.as_option.debug_dir = os.getcwd() + '/' + debug_dir

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_bert_layers(self, batch_size, seq_len, num_layers, hidden_size,
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

            if (self.as_option.mode == "profile"):
                loss_func = manual_layer_construction(loss_func)
                grads = alpa.grad(loss_func)(state.params)
            else:
                grads = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        hidden_states = jnp.ones((batch_size, seq_len, hidden_size),
                                 dtype=self.data_type)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        label = jnp.ones((batch_size, seq_len, hidden_size),
                         dtype=self.data_type)

        model = FlaxBertLayerCollection(
            BertConfig(
                num_hidden_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 4,
                num_attention_heads=num_heads,
                gradient_checkpointing=use_remat,
                add_manual_pipeline_markers=(self.as_option.mode == "profile")),
            dtype=self.data_type,
        )
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
        return state

    def test_bert_layer_1d_mesh_pb(self, mode):
        self.as_option.mode = mode
        self.as_option.pb_config.forward_dot_count = 6 * self.num_layers
        state = self.run_bert_layers(self.batch_size, self.model_config.seq_len,
                                     self.num_layers,
                                     self.model_config.hidden_size,
                                     self.model_config.num_heads, False, False,
                                     self.device_mesh_1d)


def bert_pb_analysis_profiling_passes(mode,
                                      config,
                                      batch_size,
                                      num_layer,
                                      device_num,
                                      data_type,
                                      parallel_block_opt,
                                      enum_and_bench,
                                      debug_dir=""):
    bert_test = AutoShardingAttentionTest()
    bert_test.setUp(config, batch_size, num_layer, device_num, data_type,
                    debug_dir)
    bert_test.test_bert_layer_1d_mesh_pb(mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="analysis")
    parser.add_argument("--config",
                        type=str,
                        default="C",
                        help="GPT model configuration (A, B, C, ... I)")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size for training")
    parser.add_argument("--num_layer",
                        type=int,
                        default=2,
                        help="Number of hidden layers")
    parser.add_argument("--device_num",
                        type=int,
                        default=4,
                        help="Number of devices to use")
    parser.add_argument("--data_type",
                        type=str,
                        default="float32",
                        choices=["float32", "float16"],
                        help="Data type")
    parser.add_argument("--parallel_block_opt",
                        action="store_true",
                        help="Enable parallel block optimization")
    parser.add_argument("--debug_dir",
                        type=str,
                        default="tmp",
                        help="Directory for debugging output")

    parser.add_argument("--enum_and_bench",
                        action="store_true",
                        help="benchmark compostion of all ParallelBlock.")
    parser.add_argument("--dump_comm_volume",
                        type=str,
                        choices=["true", "false"],
                        default="false",
                        help="Enable/disable DUMP_COMM_VOLUME")
    args = parser.parse_args()
    os.environ["DUMP_COMM_VOLUME"] = args.dump_comm_volume

    data_type = jnp.float32 if args.data_type == "float32" else jnp.float16
    if args.mode == "analysis":
        print("============" * 10)
        print(
            f"============ BERT test: parallel_block_opt:{args.parallel_block_opt} config:{args.config} batchsize:{args.batch_size} layers:{args.num_layer} devices:{args.device_num} ============"
        )
    bert_pb_analysis_profiling_passes(args.mode, args.config, args.batch_size,
                                      args.num_layer, args.device_num,
                                      data_type, args.parallel_block_opt,
                                      args.enum_and_bench, args.debug_dir)
