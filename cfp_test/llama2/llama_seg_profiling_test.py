'''Test auto sharding on llama2 model layers.'''

import unittest
import os
import jax

import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import optax
import jax.random as rand

import alpa
from alpa import parallelize, ShardParallel, LocalPhysicalDeviceMesh, AutoShardingOption
from llama_2_jax.lib.llama.decoder import forward_decoder, init_decoder
from llama_2_jax.lib.llama.decoder_block import DecoderBlock as Decoder, DecoderBlock, forward_decoder_block, init_decoder_block
from llama_2_jax.lib.llama.ModelConfig import ModelConfig
from llama_2_jax.lib.loss import cross_entropy_loss
from alpa.pipeline_parallel.layer_construction import manual_layer_construction
from alpa.device_mesh import VirtualPhysicalMesh
from alpa.shard_parallel.pb_config import PBConfig, default_pb_config
from dataclasses import replace
import argparse


def create_llama2_config(n_layers, d_model):
    config = ModelConfig(
        d_ff=11008,
        d_k=128,
        d_model=d_model,
        d_v=128,
        dropout_rate=0.1,
        n_heads_kv=32,
        n_layers=n_layers,  #default 32
        n_rep_kv=1,
        rms_norm_eps=1e-6,
        token_id_bos=1,
        token_id_eos=2,
        token_id_pad=0,
        vocab_size=32000,
    )
    return config


class AutoShardingLlamaTest(unittest.TestCase):

    def setUp(self,
              model_setting,
              batch_size,
              num_layers,
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
        self.model_config = create_llama2_config(num_layers, model_setting)
        self.lm_head = jnp.ones(
            [self.model_config.d_model, self.model_config.d_model])
        self.batch_size = batch_size
        self.as_option = AutoShardingOption()
        self.as_option.mode = "default"
        self.as_option.batch_size = batch_size
        self.as_option.num_layers = num_layers
        self.as_option.force_batch_dim_to_mesh_dim = -1
        self.as_option.pb_config = default_pb_config
        if (debug_dir != ""):
            self.as_option.print_strategy = True
            self.as_option.debug_dir = os.getcwd() + '/' + debug_dir

    def get_device_mesh(self, shape, mesh_alpha, mesh_beta):
        return self.physical_mesh.get_logical_mesh(shape, mesh_alpha, mesh_beta)

    def dummy_input_for_decoder(self, model_config):
        seq = jnp.ones(
            [self.batch_size, model_config.d_k, model_config.d_model],
            dtype=self.data_type)
        seq_mask = seq
        attn_mask = seq_mask
        attn_mask = jnp.tril(jnp.einsum('bik,bjk->bij', attn_mask,
                                        attn_mask))[:, None, None]
        labels = jnp.ones([self.batch_size, model_config.d_k], dtype=jnp.int32)
        labels_mask = jnp.zeros([self.batch_size, model_config.d_k],
                                dtype=jnp.int32)
        return (seq, seq_mask, labels, labels_mask, attn_mask)

    def run_llama_layers(self):

        @parallelize(method=ShardParallel(devices=self.device_mesh_1d,
                                          auto_sharding_option=self.as_option))
        def train_step(params: Decoder, opt_state, data_batch, key):

            def train_forward(params: Decoder, opt_state, data_batch, key):
                seq, seq_mask, labels, labels_mask, attn_mask = data_batch
                key, subkey = rand.split(key)
                seq_ = forward_decoder(params,
                                       seq,
                                       attn_mask,
                                       key=key,
                                       model_config=self.model_config,
                                       add_manual_pipeline_marker=(
                                           self.as_option.mode == "profile"))
                loss = cross_entropy_loss(seq_, labels, mask=labels_mask)
                return loss

            if (self.as_option.mode == "profile"):
                train_forward = manual_layer_construction(train_forward)
                grads = alpa.grad(train_forward)(params, opt_state, data_batch,
                                                 key)
            else:
                grads = jax.grad(train_forward)(params, opt_state, data_batch,
                                                key)
            updates, opt_state = optimize(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return opt_state

        key = rand.PRNGKey(20)
        key0, key1 = rand.split(key)
        params = init_decoder(key=key1, model_config=self.model_config)
        params = jax.tree_map(lambda x: x.astype(self.data_type), params)

        optimizer = optax.adamw(1e-2)
        optimize = optimizer.update
        opt_state = optimizer.init(params)

        data_batch = self.dummy_input_for_decoder(self.model_config)
        # input data shape
        opt_state = train_step(params, opt_state, data_batch, key)

    def test_llama_1d(self, mode):
        self.as_option.mode = mode
        self.as_option.pb_config.forward_dot_count = 9 * self.model_config.n_layers
        self.run_llama_layers()


def llama_pb_analysis_profiling_passes(mode, model_setting, batch_size,
                                       num_layers, device_num, data_type,
                                       parallel_block_opt, enum_and_bench,
                                       debug_dir):
    llama_test = AutoShardingLlamaTest()
    llama_test.setUp(model_setting, batch_size, num_layers, device_num,
                     data_type, debug_dir)
    llama_test.test_llama_1d(mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", type=str, default="analysis")
    parser.add_argument("--d_model",
                        type=int,
                        default=4096,
                        help="llama model hiddensize ")
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
            f"============ LLAMA2 test: parallel_block_opt:{args.parallel_block_opt} config:{args.d_model} batchsize:{args.batch_size} layers:{args.num_layer} devices:{args.device_num} ============"
        )
    llama_pb_analysis_profiling_passes(args.mode, args.d_model, args.batch_size,
                                       args.num_layer, args.device_num,
                                       data_type, args.parallel_block_opt,
                                       args.enum_and_bench, args.debug_dir)
