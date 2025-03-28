"""Test auto sharding with MoE."""

import unittest
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

import alpa
from alpa import parallelize, ShardParallel, LocalPhysicalDeviceMesh, AutoShardingOption
from alpa.model.moe import FlaxMoELayer, FlaxMoELayerCollection, FlaxMoEForLMModule, MoEConfig, TrainState
from alpa.device_mesh import VirtualPhysicalMesh
from alpa.shard_parallel.pb_config import PBConfig, default_pb_config
from alpa.pipeline_parallel.layer_construction import manual_layer_construction
from dataclasses import replace
import argparse

MoEModelConfig = namedtuple("MoEModelConfig", [
    "seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size",
    "num_experts", "expert_group_size"
])

moe_specs = {
    #                      S,    H,   L, head, V,   E,  S_
    "A": MoEModelConfig(1024, 768, 8, 16, 32000, 8, 2048),  # 380M
    "B": MoEModelConfig(1024, 768, 8, 16, 32000, 16, 2048),  # 690M
    "C": MoEModelConfig(1024, 768, 16, 16, 32000, 16, 2048),  # 1.3B
    "D": MoEModelConfig(1024, 1024, 16, 16, 32000, 16, 2048),  # 2.4B
    "E": MoEModelConfig(1024, 1280, 16, 16, 32000, 32, 2048),  # 7.1B
    "F": MoEModelConfig(1024, 1536, 16, 16, 32000, 32, 2048),  # 10B
    "G": MoEModelConfig(1024, 2048, 16, 16, 32000, 48, 2048),  # 27B
    "H": MoEModelConfig(1024, 2048, 32, 16, 32000, 64, 2048),  # 70B
    "I": MoEModelConfig(1024, 2048, 32, 16, 32000, 128, 2048),  # 140B
    "T": MoEModelConfig(1024, 768, 8, 16, 32000, 8, 2048)
}


class AutoShardingMoETest(unittest.TestCase):

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
        self.model_config = moe_specs[config_str]
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.seq_len = self.model_config.seq_len
        self.hidden_size = self.model_config.hidden_size
        self.num_heads = self.model_config.num_heads
        self.S = self.model_config.expert_group_size
        self.E = self.model_config.num_experts
        mesh_shape_1d = [1, device_num]
        self.device_mesh_1d = self.get_device_mesh(mesh_shape_1d, [1, 1],
                                                   [1, 1])
        mesh_shape_2d = [2, device_num // 2]
        self.device_mesh_2d = self.get_device_mesh(mesh_shape_2d, [1, 1],
                                                   [1, 1])
        self.as_option = AutoShardingOption()
        self.as_option.mode = "default"
        self.as_option.num_layers = num_layers // 2
        self.as_option.force_batch_dim_to_mesh_dim = -1
        self.as_option.pb_config = default_pb_config
        self.data_type = data_type
        if (debug_dir != ""):
            self.as_option.print_strategy = True
            self.as_option.debug_dir = os.getcwd() + '/' + debug_dir
            print(self.as_option.debug_dir)

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_moe_layer(self, batch_size, seq_len, hidden_size, num_heads, S, E,
                      deterministic, device_mesh):

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
                return jnp.mean((out - batch["labels"])**2)

            if (self.as_option.mode == "profile"):
                loss_func = manual_layer_construction(loss_func)
                grads = alpa.grad(loss_func)(state.params)
            else:
                grads = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        dtype = self.data_type
        hidden_states = jnp.ones((batch_size, seq_len, hidden_size),
                                 dtype=dtype)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        labels = jnp.ones((batch_size, seq_len, hidden_size), dtype=dtype)

        model = FlaxMoELayerCollection(MoEConfig(
            hidden_size=hidden_size,
            num_hidden_layers=self.num_layers,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            expert_group_size=S,
            expert_number=E,
            add_manual_pipeline_markers=(self.as_option.mode == "profile")),
                                       dtype=dtype)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, hidden_states, attention_mask)
        tx = optax.adam(1e-2)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None)

        # JIT compile
        state = train_step(
            state, {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "labels": labels,
                "rng": rngkey
            }, deterministic)

        # Get optimized HLO IR
        executable = train_step.get_last_executable()
        return state

    def test_moe_layer_1d_mesh_pb(self):
        state = self.run_moe_layer(self.batch_size, self.seq_len,
                                   self.hidden_size, self.num_heads, self.S,
                                   self.E, True, self.device_mesh_1d)

    def get_enum_bench_config(self):
        self.as_option.mode = "benchmark"
        self.as_option.pb_config.force_search_space = [2, 3, 2, 4, 2, 1, 1, 1]
        self.as_option.pb_config.forward_dot_count = 15 * (self.num_layers // 2)


def moe_pb_analysis_profiling_passes(config,
                                     batch_size,
                                     num_layers,
                                     device_num,
                                     data_type,
                                     parallel_block_opt,
                                     enum_and_bench,
                                     debug_dir=""):
    moe_test = AutoShardingMoETest()
    moe_test.setUp(config_str=config,
                   batch_size=batch_size,
                   num_layers=num_layers,
                   device_num=device_num,
                   data_type=data_type,
                   debug_dir=debug_dir)
    moe_test.get_enum_bench_config()
    moe_test.test_moe_layer_1d_mesh_pb()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--config",
                        type=str,
                        default="C",
                        help="MoE model configuration (A, B, C, ... )")
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
    print(
        f"============ MoE test: parallel_block_opt:{args.parallel_block_opt} config:{args.config} batchsize:{args.batch_size} layers:{args.num_layer} devices:{args.device_num} ============"
    )
    moe_pb_analysis_profiling_passes(args.config, args.batch_size,
                                     args.num_layer, args.device_num, data_type,
                                     args.parallel_block_opt,
                                     args.enum_and_bench, args.debug_dir)
