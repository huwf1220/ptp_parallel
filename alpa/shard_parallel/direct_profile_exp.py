"""Compile executables for shard parallelism."""
import hashlib
import inspect
from typing import Callable, Sequence, Optional, Union
import copy
import csv

import subprocess
import signal
import os
import shutil

import jax
import numpy as np
from jax import linear_util as lu
from jax._src import traceback_util
from jax._src.lib import xla_extension as xe
from jax.core import (Jaxpr, ClosedJaxpr, Literal, gensym, get_aval,
                      raise_to_shaped, AbstractValue)
from jax.lax import add_p, div_p
from jax.tree_util import PyTreeDef

from alpa.device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh, VirtualPhysicalMesh, LocalPhysicalDeviceMesh
from alpa.global_env import global_config, parallel_compile_config
from alpa.mesh_executable import (NormalMeshDriverExecutable,
                                  GradAccMeshDriverExecutable)
from alpa.pipeline_parallel.apply_grad import APPLY_GRAD_MARKER_SUFFIX
from alpa.shard_parallel.auto_sharding import (run_auto_sharding_pass,
                                               run_spmd_partitioner_pass,
                                               AutoShardingOption)
from alpa.util import (jaxpr_to_hlo, trace_jaxpr_with_micro_batch,
                       setup_computation_alias, OrderedSet, new_jaxpr_eqn)

from jax._src.lib import xla_bridge as xb, xla_extension as xe
import time
from tensorflow.compiler.xla.service import reindexing_pb2
import multiprocessing
import tensorflow as tf

from alpa.shard_parallel.pb_config import ProfileConfig
from alpa.pipeline_parallel.computation import slice_closed_jaxpr_by_full_pipeline_marks

traceback_util.register_exclusion(__file__)


def enumerate_lists(arr: Sequence[int]):
    n = len(arr)
    result = []
    for i in range(arr[0]):
        if n == 1:
            result.append([i])
        else:
            sub_lists = enumerate_lists(arr[1:])
            for sub_list in sub_lists:
                result.append([i] + sub_list)
    return result


def enumerate_strategy_combination(search_stra_num: Sequence[int],
                                   search_file_path):
    result = enumerate_lists(search_stra_num)
    file = open(search_file_path, 'w')
    for stra_vec in result:
        line = ','.join(str(num) for num in stra_vec)
        file.write(line + '\n')
    file.close()
    return


def delete_profile_files(path):
    shutil.rmtree(path)


def benchmark_func(sharded_exec, warmup=10, repeat=3, number=10):
    costs = []
    tic = time.time()
    for _ in range(number):
        sharded_exec()
    c = (time.time() - tic) / number
    for _ in range(repeat):
        tic = time.time()
        for _ in range(number):
            sharded_exec()
        costs.append(time.time() - tic)
    if (costs[repeat - 1] / number) < parallel_compile_config.best_cost:
        parallel_compile_config.best_cost = costs[repeat - 1] / number
    #jax.profiler.start_trace("./tmp")
    sharded_exec()
    #jax.profiler.stop_trace()
    #delete_profile_files("./tmp")
    return np.array(costs) / number


def benchamrk_strategy(compiled, local_devices, device_inputs):
    costs = []

    def sharded_exec():
        device_outputs = compiled.execute_sharded_on_local_devices(
            device_inputs)
        ct = 0
        for j in range(len(device_inputs)):
            if device_inputs[j][0].is_deleted():
                device_inputs[j] = device_outputs[ct]
                ct += 1
        local_devices[0].synchronize_all_activity()

    try:
        costs = benchmark_func(sharded_exec)
    except RuntimeError:
        costs = [0 for _ in range(3)]
    return costs


def test_exec(exec, pb_val):
    hlo_module = exec.compiled.hlo_modules()[0]
    local_devices = exec.physical_mesh.devices
    cost_failed = [-1, -1, -1]
    input_shapes = hlo_module.parameter_shapes()
    # prune OOM cases, not exact because third party lib not considered:
    free_mem = local_devices[0].available_memory()
    if free_mem < exec.compiled.total_allocation_size() and free_mem != -1:
        print("OOM")
        return cost_failed
    device_inputs = []
    try:
        for shape in input_shapes:
            device_inputs.append([
                exec.physical_mesh.backend.buffer_from_pyval(
                    np.empty(shape.dimensions(), shape.numpy_dtype()), device)
                for device in local_devices
            ])
        local_devices[0].synchronize_all_activity()
    except RuntimeError:
        costs = cost_failed
    costs = benchamrk_strategy(exec.compiled, local_devices, device_inputs)

    print("[PROFILED] PB_val:{}, costs:{}".format(pb_val, costs))
    return costs


def _clear_backend_set_default_allocator(physical_mesh):
    if (os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] != "platform"):
        print(
            "Can Only Free GPU Memory For Delete Buffers That Allocated By \"platform\" Allocators."
        )
    for buf in physical_mesh.backend.live_buffers():
        buf.delete()
    physical_mesh.shutdown()
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
    jax.clear_backends()


'''
def worker_backend_compile(
            queue, # a queue to let subprocess send signal to main process when it profiling is done.
            events,# set next subprocess to start profiling.
            rank,
            pb_vals,
            init_hlo,
            as_option: AutoShardingOption,
            stage_plan,
            avals,
            out_avals,
            donated_invars,
            static_argnums: Sequence[int]):
    parallel_compile_config.set_cpu_affinity(rank)
    # reinit a LocalPhysicalDeviceMesh for subprocess
    physical_mesh = LocalPhysicalDeviceMesh()
    logical_mesh_choices = [physical_mesh.get_logical_mesh()]
    execs = []
    compiled_pbvals = []
    print("[worker {}]: start compiling {} execs.".format(rank, len(pb_vals)))
    for pb_val in pb_vals:
        pb_val = pb_val  # [FIXME]: 4 layers
        sval = pb_val
        hlo_ = copy.copy(init_hlo)
        flop_count = xe.hlo_module_count_flop_dot_conv_only(hlo_.get_module())
        hlo_, stage_plan = run_auto_sharding_pass(hlo_, logical_mesh_choices[0],
                                                 "single", 1, as_option, solution_vector = sval) 

        hlo_ = run_spmd_partitioner_pass(hlo_, np.prod(logical_mesh_choices[0].shape))
        if isinstance(physical_mesh, VirtualPhysicalMesh):
            raise NotImplementedError()

        process_id = os.getpid()
        failed_compile_flag = ""
        flag_file_name = f"/tmp/failed_compile_{process_id}.txt"       #TODO: replace this with simple run_backend_compile() as we just want the compiled code.
       

        exec = NormalMeshDriverExecutable(physical_mesh,
                                  hlo_,
                                  stage_plan,
                                  avals,
                                  out_avals,
                                  donated_invars,
                                  static_argnums=static_argnums,
                                  in_tree=None,
                                  out_tree=None,
                                  flop_count=flop_count)

        if os.path.isfile(flag_file_name):
            with open(flag_file_name, 'r') as flag_file:
                failed_compile_flag = flag_file.read().strip()
        if failed_compile_flag == 'True':
            print("failed compiled {}.".format(pb_val))
            os.remove(flag_file_name)
            continue
        else:
            execs.append(exec)   
            compiled_pbvals.append(pb_val)
            print("successful compiled {}.".format(pb_val))
    print("[worker {}]: end compiling execs.".format(rank))
    # start profile
    while True:
        events[rank].wait()
        events[rank].clear()
        _clear_backend_set_default_allocator(physical_mesh) 
        physical_mesh = LocalPhysicalDeviceMesh()
        #time.sleep(5)
        #print("[worker {}]: start profile {} execs.".format(rank, len(execs)))
        count = 0
        for exec in execs:
            exec.physical_mesh = physical_mesh
            exec.compiled.change_client(physical_mesh.backend)
            cost = test_exec(exec, compiled_pbvals[count])
            count = count + 1
        break
    queue.put(os.getpid())
    events[rank+1].set()
'''


def read_reindex_items_list(file_path):
    reindex_items_list = []
    with open(file_path, "rb") as f:
        reindex_list = reindexing_pb2.ReindexItemList()
        reindex_list.ParseFromString(f.read())
        for reindex_item in reindex_list.reindex_items:
            reindex_items_list.append({
                "inst_leaf_id":
                    reindex_item.inst_leaf_id,
                "inst_id":
                    reindex_item.inst_id,
                "stra_num":
                    reindex_item.stra_num,
                "deps_by_multi_pb":
                    reindex_item.deps_by_multi_pb,
                "is_key_inst":
                    reindex_item.is_key_inst,
                "deps_pb_id":
                    list(reindex_item.deps_pb_id),
                "reindexing": [
                    list(reindex_data.reindex)
                    for reindex_data in reindex_item.reindexing
                ],
            })
    return reindex_items_list


def check_reindex_list(reindex_list):
    '''
    TODO[huwf] add a function to check if each reindex item belongs 
    {
      1.key_inst flag, 
      2.multi_pb flag,
      3.none of these two flag and have exactly one pb user and reindex, 
      4.stra_num==1, 
    }
    '''
    search_space = []
    for item in reindex_list:
        if item['stra_num'] > 1:
            #print("INST {} deps_id is {} is_key_inst: {} deps_by_multi_pb {} stra_num {} len of reindexing is {} ".format(item['inst_id'], item['deps_pb_id'], item['is_key_inst'], item['deps_by_multi_pb'], item['stra_num'], len(item['reindexing'])))
            pass
        if (item['is_key_inst']):
            #print(f"Key inst: {item['inst_id']}")
            search_space.append(item['stra_num'])
        elif (item['stra_num'] == 1 or item['deps_by_multi_pb']):
            pass
        elif (not item['deps_by_multi_pb'] and not item['is_key_inst'] and
              item['stra_num'] != 1):
            assert len(
                item['reindexing']
            ) > 0, "Wrong reindexing for followed insts! INST {}".format(
                item['inst_id'])
        else:
            assert 1, "Cannot handle this inst."
    return search_space


def shard_parallel_internal_exp(
        fun: lu.WrappedFun, in_tree: PyTreeDef, out_tree_thunk: Callable,
        static_argnums: Sequence[int], donated_invars: Sequence[bool],
        physical_mesh: PhysicalDeviceMesh,
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        as_option: AutoShardingOption, *avals: Sequence[AbstractValue]):

    print("****************************************")
    print(as_option.profile_config)
    print("****************************************")
    _debug_dir = os.getcwd(
    ) + '/tmp' if as_option.debug_dir == '' else as_option.debug_dir
    if not os.path.exists(_debug_dir):
        os.makedirs(_debug_dir)
    # pylint: disable=unused-argument
    # Trace to get jaxpr
    closed_jaxpr, _ = trace_jaxpr_with_micro_batch(fun, [False] * len(avals), 1,
                                                   avals)

    out_avals = [v.aval for v in closed_jaxpr.jaxpr.outvars]
    name = f"{fun.__name__}_shard_parallel"
    init_hlo = jaxpr_to_hlo(name, closed_jaxpr, donated_invars)
    hlo = copy.copy(init_hlo)
    flop_count = xe.hlo_module_count_flop_dot_conv_only(hlo.get_module())
    hlo_ = copy.copy(init_hlo)

    as_option.profile_config.profile_mode = False
    as_option.print_strategy = True
    as_option.debug_dir = _debug_dir
    # run default auto-sharding to get following relations -> reindexing list.
    hlo_, stage_plan = run_auto_sharding_pass(hlo_, logical_mesh_choices[0],
                                              "single", 1, as_option)
    as_option.print_strategy = True

    if global_config.backend == "gpu":
        hlo_ = run_spmd_partitioner_pass(hlo_,
                                         np.prod(logical_mesh_choices[0].shape))
    if isinstance(physical_mesh, VirtualPhysicalMesh):
        raise NotImplementedError()
        return None
    # Compile a mesh executable
    exec = NormalMeshDriverExecutable(physical_mesh,
                                      hlo_,
                                      stage_plan,
                                      avals,
                                      out_avals,
                                      donated_invars,
                                      static_argnums=static_argnums,
                                      in_tree=in_tree,
                                      out_tree=out_tree_thunk(),
                                      flop_count=flop_count)
    #_clear_backend_set_default_allocator(physical_mesh)
    physical_mesh = LocalPhysicalDeviceMesh()
    exec.physical_mesh = physical_mesh
    exec.compiled.change_client(physical_mesh.backend)
    default_cost = test_exec(exec, [-1, -1, -1])
    if not as_option.parallel_block_opt:
        return exec
    # -----------------------------------------------------------------------------------------
    # ----------------------------- profile configuration -------------------------------------
    as_option.profile_config.profile_mode = True
    reindex_list = read_reindex_items_list(as_option.debug_dir +
                                           "/reindex_array")
    generated_search_space = check_reindex_list(reindex_list)
    print(generated_search_space)

    target_stras = []

    force_search_space = as_option.profile_config.force_search_space
    force_pb_lists = as_option.profile_config.force_pb_lists
    if (len(force_search_space) != 0):
        search_space_ = force_search_space
        pbval_list = enumerate_lists(search_space_)
        if (len(force_pb_lists) != 0):
            pbval_list = force_pb_lists
        for lis in pbval_list:
            target_stras.append(lis * as_option.profile_config.seg_num)

    # -----------------------------------------------------------------------------------------
    # ----------------------------- profiling start -------------------------------------

    for pb_val in target_stras:
        sval = pb_val
        hlo_ = copy.copy(init_hlo)
        flop_count = xe.hlo_module_count_flop_dot_conv_only(hlo_.get_module())
        hlo_, stage_plan = run_auto_sharding_pass(hlo_,
                                                  logical_mesh_choices[0],
                                                  "single",
                                                  1,
                                                  as_option,
                                                  solution_vector=sval)

        hlo_ = run_spmd_partitioner_pass(hlo_,
                                         np.prod(logical_mesh_choices[0].shape))
        if isinstance(physical_mesh, VirtualPhysicalMesh):
            raise NotImplementedError()

        exec = NormalMeshDriverExecutable(physical_mesh,
                                          hlo_,
                                          stage_plan,
                                          avals,
                                          out_avals,
                                          donated_invars,
                                          static_argnums=static_argnums,
                                          in_tree=None,
                                          out_tree=None,
                                          flop_count=flop_count)
        physical_mesh = LocalPhysicalDeviceMesh()
        exec.physical_mesh = physical_mesh
        exec.compiled.change_client(physical_mesh.backend)
        cost = test_exec(exec, pb_val)

    return exec
