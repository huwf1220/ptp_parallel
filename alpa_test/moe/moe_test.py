"""Test auto sharding with MoE."""

import unittest
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

from alpa import parallelize, ShardParallel, LocalPhysicalDeviceMesh, AutoShardingOption
from alpa.model.moe import FlaxMoELayer, FlaxMoELayerCollection, FlaxMoEForLMModule, MoEConfig, TrainState
from alpa.device_mesh import VirtualPhysicalMesh
import argparse

MoEModelConfig = namedtuple("MoEModelConfig", [
    "seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size",
    "num_experts", "expert_group_size"
])

moe_specs = {
    #                      S,    H,   L, head, V,   E,  S_
    "A": MoEModelConfig(1024, 768, 8, 16, 32000, 8, 2048), # 380M
    "B": MoEModelConfig(1024, 768, 8, 16, 32000, 16, 2048),# 690M
    "C": MoEModelConfig(1024, 768, 16, 16, 32000, 16, 2048), # 1.3B
    "D": MoEModelConfig(1024, 1024, 16, 16, 32000, 16, 2048), # 2.4B
    "E": MoEModelConfig(1024, 1280, 16, 16, 32000, 32, 2048), # 7.1B
    "F": MoEModelConfig(1024, 1536, 16, 16, 32000, 32, 2048), # 10B
    "G": MoEModelConfig(1024, 2048, 16, 16, 32000, 48, 2048), # 27B
    "H": MoEModelConfig(1024, 2048, 32, 16, 32000, 64, 2048), # 70B
    "I": MoEModelConfig(1024, 2048, 32, 16, 32000, 128, 2048) # 140B
}

class AutoShardingMoETest(unittest.TestCase):
    def setUp(self, config_str="A", batch_size=32, num_layers=1, device_num=4, data_type=jnp.float32, debug_dir=""):
        if len(jax.local_devices()) >= device_num:
            self.physical_mesh = LocalPhysicalDeviceMesh(
                jax.local_devices()[:device_num])
        else:  # only for test compile and debug.
            self.physical_mesh = VirtualPhysicalMesh(
                ["host_0"], [{"NodeManagerAddress": "1.1.1.1"}], 8)
        self.model_config = moe_specs[config_str]
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.data_type = data_type
        self.seq_len = self.model_config.seq_len
        self.hidden_size = self.model_config.hidden_size
        self.num_heads = self.model_config.num_heads
        self.S = self.model_config.expert_group_size
        self.E = self.model_config.num_experts
        mesh_shape_1d = [1, device_num]
        self.device_mesh_1d = self.get_device_mesh(
            mesh_shape_1d, [1, 1], [1, 1])
        mesh_shape_2d = [2, device_num // 2]
        self.device_mesh_2d = self.get_device_mesh(
            mesh_shape_2d, [1, 1], [1, 1])
        self.as_option = AutoShardingOption()
        self.as_option.force_batch_dim_to_mesh_dim = -1
        if (debug_dir != ""):
            self.as_option.print_strategy = True
            self.as_option.debug_dir = os.getcwd()+'/'+debug_dir

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def run_moe(self, batch_size, seq_len, hidden_size, num_heads, S, E,
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

            grads = jax.grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        dtype = self.data_type
        hidden_states = jnp.ones((batch_size, seq_len, hidden_size),
                                 dtype=dtype)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        labels = jnp.ones((batch_size, seq_len, hidden_size), dtype=dtype)

        # LayerCollection for multi-layers

        model = FlaxMoELayerCollection(
            MoEConfig(
            hidden_size=hidden_size,
            num_hidden_layers=self.num_layers,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            expert_group_size=S,
            expert_number=E,),
        dtype=dtype
        )
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
        return (state, executable.get_hlo_text(),
                executable.auto_sharding_objective)


    def test_moe(self):
        deterministic = True
        state, hlo_ir, objective = self.run_moe(
                self.batch_size, self.seq_len, self.hidden_size, self.num_heads, self.S, self.E,
                deterministic, self.device_mesh_1d)
    

def moe_alpa_default_perf_test(config, batch_size, num_layers, device_num, data_type, debug_dir=""):
    moe_test = AutoShardingMoETest()
    moe_test.setUp(config_str=config, batch_size=batch_size, num_layers=num_layers, device_num=device_num, data_type=data_type, debug_dir=debug_dir)
    moe_test.test_moe()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="C", help="GPT model configuration (A, B, C, ... I)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_layer", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--device_num", type=int, default=4, help="Number of devices to use")
    parser.add_argument("--data_type", type=str, default="float32", choices=["float32", "float16"], help="Data type")
    parser.add_argument("--debug_dir", type=str, default="tmp", help="Directory for debugging output")
    parser.add_argument("--dump_comm_volume", type=str, choices=["true", "false"], default="false", help="Enable/disable DUMP_COMM_VOLUME")
    args = parser.parse_args()

    os.environ["DUMP_COMM_VOLUME"] = args.dump_comm_volume

    data_type = jnp.float32 if args.data_type == "float32" else jnp.float16
    print(f"============ Alpa MoE test: config:{args.config} batchsize:{args.batch_size} layers:{args.num_layer} devices:{args.device_num} ============")
    moe_alpa_default_perf_test(args.config, args.batch_size, args.num_layer, args.device_num, data_type, args.debug_dir)

