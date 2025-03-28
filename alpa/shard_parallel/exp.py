"""Compile executables for shard parallelism."""

import time
import copy
import os
import shutil
from itertools import product
from typing import Callable, Sequence
import numpy as np

import jax
from jax import linear_util as lu
from jax._src import traceback_util
from jax.core import AbstractValue
from jax.tree_util import PyTreeDef

from alpa.device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh, VirtualPhysicalMesh, LocalPhysicalDeviceMesh
from alpa.mesh_executable import NormalMeshDriverExecutable
from alpa.shard_parallel.auto_sharding import (run_auto_sharding_pass,
                                               run_spmd_partitioner_pass,
                                               AutoShardingOption,
                                               get_input_output_sharding_specs)
from alpa.util import (jaxpr_to_hlo, trace_jaxpr_with_micro_batch)
from jax._src.lib import xla_bridge as xb, xla_extension as xe
from tensorflow.compiler.xla.service import reindexing_pb2
from collections import defaultdict
# jaxpr functions
from alpa.shard_parallel.segment_jaxpr import (segment_jaxpr_by_pp_markers,
                                               SegPbSpecs,
                                               build_space_and_search)
import copy

import logging
import sys

DEBUG_MODE = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.WARNING)

if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
logger.propagate = False

GB = 1024**3
MEMORY_QUANTIZATION_UNIT = 0.5 * GB  # 0.5 GB


def quantize_memory(mem_bytes: float) -> float:
    """Quantize memory usage, return x.x GBs"""
    return round(
        mem_bytes / MEMORY_QUANTIZATION_UNIT) * (MEMORY_QUANTIZATION_UNIT / GB)


def dump_to_file(str, path):
    with open(path, "w") as f:
        f.write(str)
    f.close()


def enumerate_combinations(arr: Sequence[int]):
    if not arr:
        return []
    ranges = [range(n) for n in arr]
    return [list(comb) for comb in product(*ranges)]


def enumerate_strategy_combination(search_stra_num: Sequence[int],
                                   search_file_path):
    """Enumerate all strategy combinations and save to file."""
    combinations = enumerate_combinations(search_stra_num)
    with open(search_file_path, 'w') as f:
        f.writelines(','.join(map(str, vec)) + '\n' for vec in combinations)


def delete_profile_files(path):
    shutil.rmtree(path)


def benchmark_func(sharded_exec,
                   curr_optimal=float('inf'),
                   warmup=10,
                   repeat=3,
                   number=100,
                   dump_iters=0,
                   keep_trace_file=False):
    costs = []
    tic = time.time()
    for _ in range(2):
        sharded_exec()
    c = (time.time() - tic) / 2
    if (c > 1.5 * curr_optimal):
        return np.array([c for _ in range(repeat)])
    for _ in range(repeat):
        tic = time.time()
        for _ in range(number):
            sharded_exec()
        costs.append(time.time() - tic)

    if dump_iters <= 0:
        return np.array(costs) / number
    jax.profiler.start_trace("./tmp_trace")
    for _ in range(dump_iters):
        sharded_exec()
    jax.profiler.stop_trace()
    if not keep_trace_file:
        delete_profile_files("./tmp_trace")
    return np.array(costs) / number


def benchmark_strategy(compiled,
                       local_devices,
                       device_inputs,
                       curr_optimal=None,
                       profile_iters=10,
                       dump_iters=0):
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

    if curr_optimal is None:
        curr_optimal = float('inf')
    try:
        costs = benchmark_func(sharded_exec,
                               curr_optimal,
                               number=profile_iters,
                               dump_iters=dump_iters)
    except RuntimeError:
        costs = [-1 for _ in range(3)]
    return costs


def test_exec(exec,
              pb_val,
              mode,
              curr_optimal=None,
              force_profile_iters=None,
              dump_iters=0):
    hlo_module = exec.compiled.hlo_modules()[0]
    local_devices = exec.physical_mesh.devices
    cost_failed = [-1, -1, -1]
    input_shapes = hlo_module.parameter_shapes()
    # prune OOM cases, not exact because third party lib not considered:

    free_mem = local_devices[0].available_memory()
    # logger.info(f"free_mem: {free_mem}")
    if free_mem < exec.compiled.total_allocation_size() and free_mem != -1:
        # logger.info(f"need: {exec.compiled.total_allocation_size()}")
        if (force_profile_iters is not None and force_profile_iters < 10):
            return cost_failed
        logging.warning(
            f"OOM: free_mem: {free_mem} need: {exec.compiled.total_allocation_size()}"
        )
        return cost_failed
    # logger.info(f"need: {exec.compiled.total_allocation_size()}")
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
    profile_iters = 10
    if mode == "default" or mode == "search":
        profile_iters = 100  # 100
    if force_profile_iters is not None:
        profile_iters = force_profile_iters
    costs = benchmark_strategy(exec.compiled,
                               local_devices,
                               device_inputs,
                               curr_optimal,
                               profile_iters=profile_iters,
                               dump_iters=dump_iters)
    mem_cost = quantize_memory(exec.compiled.total_allocation_size())
    if mode == "default":
        logger.info(f"default training cost: {costs}, memory_cost: {mem_cost}")
    elif mode == "search":
        if force_profile_iters >= 10:
            logger.info(
                f"searched training cost: {costs}, memory_cost: {mem_cost}")
    elif mode == "analysis_global":
        logger.info(f"analysis training cost: {costs}, memory_cost: {mem_cost}")
    else:
        pass

    return costs


def _clear_backend_set_default_allocator(physical_mesh):
    if (os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] != "platform"):
        logging.warning(
            "Can Only Free GPU Memory For Delete Buffers That Allocated By \"platform\" Allocators."
        )
    for buf in physical_mesh.backend.live_buffers():
        buf.delete()
    physical_mesh.shutdown()
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
    jax.clear_backends()


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
            #logging.debug("INST {} deps_id is {} is_key_inst: {} deps_by_multi_pb {} stra_num {} len of reindexing is {} ".format(item['inst_id'], item['deps_pb_id'], item['is_key_inst'], item['deps_by_multi_pb'], item['stra_num'], len(item['reindexing'])))
            pass
        if (item['is_key_inst']):
            #logging.debug(f"Key inst: {item['inst_id']}")
            search_space.append(item['stra_num'])
        elif (item['stra_num'] == 1 or item['deps_by_multi_pb']):
            pass
        elif (not item['deps_by_multi_pb'] and not item['is_key_inst'] and
              item['stra_num'] != 1):
            if item['inst_id'] == 0:
                continue
            assert len(
                item['reindexing']
            ) > 0, "Wrong reindexing for followed insts! INST {}".format(
                item['inst_id'])
        else:
            assert 1, "Cannot handle this inst."
    return search_space


def save_profiles(segment_profiling_data, unique_seg_deps, segs_pb_specs):
    import pickle

    data_dict = {
        "segment_profiling.pkl": segment_profiling_data,
        "unique_seg_deps.pkl": unique_seg_deps,
        "segs_pb_specs.pkl": segs_pb_specs,
    }

    for filename, data in data_dict.items():
        with open(filename, "wb") as f:
            pickle.dump(data, f)


def load_profiles():
    import pickle
    try:
        with open("segment_profiling.pkl", "rb") as f:
            segment_profiling_data = pickle.load(f)
        with open("unique_seg_deps.pkl", "rb") as f:
            unique_seg_deps = pickle.load(f)
        with open("segs_pb_specs.pkl", "rb") as f:
            segs_pb_specs = pickle.load(f)
    except FileNotFoundError as e:
        logging.warning(f"Error: {e}")
        return None, None, None
    except pickle.UnpicklingError as e:
        logging.warning(f"Error loading pickle file: {e}")
        return None, None, None

    return segment_profiling_data, unique_seg_deps, segs_pb_specs


def read_segment_mapping(filename):
    seg_idx = []
    dot_num = []
    with open(filename, "r") as f:
        for line in f:
            first, second = map(int, line.strip().split(","))
            seg_idx.append(first)
            dot_num.append(second)
    mapping = []
    for _ in range(max(seg_idx) + 1):
        mapping.append([])
    for idx, seg_id in enumerate(seg_idx):
        mapping[seg_id].append(idx)
    return mapping, dot_num


def run_backend_compile(closed_jaxpr,
                        init_hlo,
                        logical_mesh_choice,
                        as_option,
                        physical_mesh,
                        donate_invars,
                        flop_count,
                        sval=None,
                        skip_exec_test=True,
                        seg_pb_specs=None):
    hlo = copy.copy(init_hlo)
    avals = [v.aval for v in closed_jaxpr.jaxpr.invars]
    out_avals = [v.aval for v in closed_jaxpr.jaxpr.outvars]
    (hlo, stage_plan) = run_auto_sharding_pass(hlo,
                                               logical_mesh_choice,
                                               "single",
                                               1,
                                               as_option,
                                               solution_vector=sval)
    hlo = run_spmd_partitioner_pass(hlo, np.prod(logical_mesh_choice.shape))
    if seg_pb_specs != None and sval != None:
        input_sharding_specs, output_sharding_specs = get_input_output_sharding_specs(
            hlo.get_module(), avals, out_avals,
            np.prod(logical_mesh_choice.shape), logical_mesh_choice.shape)
        invar_sharding_specs = []
        outvar_sharding_specs = []
        for invar_idx in seg_pb_specs.get_invar_list():
            invar_sharding_specs.append(input_sharding_specs[invar_idx])
        for outvar_idx in seg_pb_specs.get_outvar_list():
            outvar_sharding_specs.append(output_sharding_specs[outvar_idx])
        seg_pb_specs.add_invars_pb_and_specs(sval, invar_sharding_specs)
        seg_pb_specs.add_outvars_pb_and_specs(sval, outvar_sharding_specs)
    if skip_exec_test:
        return None, [-1, -1, -1], 1024
    exec = NormalMeshDriverExecutable(physical_mesh,
                                      hlo,
                                      stage_plan,
                                      avals,
                                      out_avals,
                                      donate_invars,
                                      static_argnums=None,
                                      in_tree=None,
                                      out_tree=None,
                                      flop_count=flop_count)
    return exec


def run_exec(exec: NormalMeshDriverExecutable,
             pb_list=[-1],
             mode="default",
             curr_optimal=None):
    mem_cost = exec.compiled.total_allocation_size()
    # #_clear_backend_set_default_allocator(physical_mesh)
    physical_mesh = LocalPhysicalDeviceMesh()
    exec.physical_mesh = physical_mesh
    exec.compiled.change_client(physical_mesh.backend)
    cost = test_exec(exec, pb_list, mode, curr_optimal, force_profile_iters=3)

    return cost, mem_cost


def run_as_exec(closed_jaxpr,
                init_hlo,
                logical_mesh_choice,
                as_option,
                physical_mesh,
                donate_invars,
                flop_count,
                sval=None,
                skip_exec_test=True,
                pb_list=[-1],
                seg_pb_specs=None,
                curr_optimal=None,
                force_profile_iters=None,
                dump_iters=0):
    hlo = copy.copy(init_hlo)
    avals = [v.aval for v in closed_jaxpr.jaxpr.invars]
    out_avals = [v.aval for v in closed_jaxpr.jaxpr.outvars]
    (hlo, stage_plan) = run_auto_sharding_pass(hlo,
                                               logical_mesh_choice,
                                               "single",
                                               1,
                                               as_option,
                                               solution_vector=sval)
    hlo = run_spmd_partitioner_pass(hlo, np.prod(logical_mesh_choice.shape))
    if seg_pb_specs != None and sval != None:
        input_sharding_specs, output_sharding_specs = get_input_output_sharding_specs(
            hlo.get_module(), avals, out_avals,
            np.prod(logical_mesh_choice.shape), logical_mesh_choice.shape)
        invar_sharding_specs = []
        outvar_sharding_specs = []
        for invar_idx in seg_pb_specs.get_invar_list():
            invar_sharding_specs.append(input_sharding_specs[invar_idx])
        for outvar_idx in seg_pb_specs.get_outvar_list():
            outvar_sharding_specs.append(output_sharding_specs[outvar_idx])
        seg_pb_specs.add_invars_pb_and_specs(sval, invar_sharding_specs)
        seg_pb_specs.add_outvars_pb_and_specs(sval, outvar_sharding_specs)

    exec = NormalMeshDriverExecutable(physical_mesh,
                                      hlo,
                                      stage_plan,
                                      avals,
                                      out_avals,
                                      donate_invars,
                                      static_argnums=None,
                                      in_tree=None,
                                      out_tree=None,
                                      flop_count=flop_count)
    if skip_exec_test:
        return exec, [-1, -1, -1], 1024
    mem_cost = exec.compiled.total_allocation_size()
    # #_clear_backend_set_default_allocator(physical_mesh)
    physical_mesh = LocalPhysicalDeviceMesh()
    exec.physical_mesh = physical_mesh
    exec.compiled.change_client(physical_mesh.backend)
    cost = test_exec(exec, pb_list, as_option.mode, curr_optimal,
                     force_profile_iters, dump_iters)
    return exec, cost, mem_cost


def prune_profile_cost(segment_profiling_data: dict, n=10) -> dict:
    """Prune profiling data keeping top 10 cost and memory entries."""
    for seg_idx, pb_dict in segment_profiling_data.items():
        items = pb_dict.items()

        top_cost = dict(sorted(items, key=lambda x: x[1][0])[:n])
        top_mem = dict(sorted(items, key=lambda x: x[1][1])[:n])

        segment_profiling_data[seg_idx] = {**top_cost, **top_mem}
    return segment_profiling_data


def overlapped_profiling(seg_idx,
                         target_stras,
                         segment_profiling_data,
                         seg_final_closed_jaxpr,
                         seg_hlo,
                         logical_mesh_choice,
                         seg_as_option,
                         physical_mesh,
                         seg_donate_invars,
                         seg_flop_count,
                         skip_exec_test=False,
                         seg_pb_specs=None,
                         dump_profiles=True):
    import threading
    import queue
    compile_queue = queue.Queue(maxsize=10)
    result_queue = queue.Queue(maxsize=10)

    def compile_worker():
        for pb_val in target_stras:
            exec_obj = run_backend_compile(seg_final_closed_jaxpr,
                                           seg_hlo,
                                           logical_mesh_choice,
                                           seg_as_option,
                                           physical_mesh,
                                           seg_donate_invars,
                                           seg_flop_count,
                                           sval=pb_val,
                                           skip_exec_test=False,
                                           seg_pb_specs=seg_pb_specs)
            compile_queue.put((pb_val, exec_obj))
        compile_queue.put(None)

    def exec_worker():
        while True:
            item = compile_queue.get()
            if item is None:
                break
            pb_val, exec_obj = item
            cost, mem_cost = run_exec(exec_obj,
                                      pb_val,
                                      mode=seg_as_option.mode,
                                      curr_optimal=global_curr_optimal)
            result_queue.put((pb_val, cost, mem_cost))
        result_queue.put(None)

    global_curr_optimal = float('inf')

    compile_thread = threading.Thread(target=compile_worker)
    exec_thread = threading.Thread(target=exec_worker)
    compile_thread.start()
    exec_thread.start()

    while True:
        result = result_queue.get()
        if result is None:
            break
        pb_val, cost, mem_cost = result
        pb_val_tuple = tuple(pb_val)
        avg_cost = (cost[0] + cost[1] + cost[2]) / 3
        if avg_cost < 0:
            avg_cost = float('inf')
        if avg_cost < global_curr_optimal:
            global_curr_optimal = avg_cost
        mem_cost_quantized = quantize_memory(mem_cost)
        if (dump_profiles):
            logger.info(
                f"SegmentIdx: {seg_idx} ParallelBlock Strategy: {pb_val} avgcost: {avg_cost} mem_cost(GB): {mem_cost_quantized}"
            )
        segment_profiling_data[seg_idx][pb_val_tuple] = (avg_cost,
                                                         mem_cost_quantized)

    compile_thread.join()
    exec_thread.join()


def shard_parallel_internal_default(
        fun: lu.WrappedFun, in_tree: PyTreeDef, out_tree_thunk: Callable,
        static_argnums: Sequence[int], donated_invars: Sequence[bool],
        physical_mesh: PhysicalDeviceMesh,
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        as_option: AutoShardingOption, *avals: Sequence[AbstractValue]):

    assert as_option.mode == "default"

    # pylint: disable=unused-argument
    # Trace to get jaxpr
    closed_jaxpr, _ = trace_jaxpr_with_micro_batch(fun, [False] * len(avals), 1,
                                                   avals)
    name = f"{fun.__name__}_shard_parallel"
    init_hlo = jaxpr_to_hlo(name, closed_jaxpr, donated_invars)
    flop_count = xe.hlo_module_count_flop_dot_conv_only(init_hlo.get_module())

    as_option.print_strategy = True
    as_option.debug_dir = f'./debug_{as_option.mode}'

    exec, cost, mem_cost = run_as_exec(closed_jaxpr,
                                       init_hlo,
                                       logical_mesh_choices[0],
                                       as_option,
                                       physical_mesh,
                                       donated_invars,
                                       flop_count,
                                       sval=None,
                                       skip_exec_test=False)
    return exec


def shard_parallel_internal_exp(
        fun: lu.WrappedFun, in_tree: PyTreeDef, out_tree_thunk: Callable,
        static_argnums: Sequence[int], donated_invars: Sequence[bool],
        physical_mesh: PhysicalDeviceMesh,
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        as_option: AutoShardingOption, *avals: Sequence[AbstractValue]):

    assert as_option.mode == "analysis"
    logger.info("***********Analysis Pass****************")

    # pylint: disable=unused-argument
    # Trace to get jaxpr
    closed_jaxpr, _ = trace_jaxpr_with_micro_batch(fun, [False] * len(avals), 1,
                                                   avals)
    name = f"{fun.__name__}_shard_parallel"
    init_hlo = jaxpr_to_hlo(name, closed_jaxpr, donated_invars)
    flop_count = xe.hlo_module_count_flop_dot_conv_only(init_hlo.get_module())

    start = time.time()
    as_option.mode = "analysis_global"
    as_option.print_strategy = True
    as_option.debug_dir = f'./debug_{as_option.mode}'

    exec, cost, mem_cost = run_as_exec(closed_jaxpr,
                                       init_hlo,
                                       logical_mesh_choices[0],
                                       as_option,
                                       physical_mesh,
                                       donated_invars,
                                       flop_count,
                                       sval=None,
                                       skip_exec_test=True)
    print(f"global analysis time: {time.time()-start}")
    print(f"FLOP_COUNT: {flop_count/1e12}")
    exit()


def shard_parallel_internal_seg_profiling(
        fun: lu.WrappedFun, in_tree: PyTreeDef, out_tree_thunk: Callable,
        static_argnums: Sequence[int], donated_invars: Sequence[bool],
        physical_mesh: PhysicalDeviceMesh,
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        as_option: AutoShardingOption, *avals: Sequence[AbstractValue]):

    assert as_option.mode == "profile"

    logger.info("***********Profiling Pass****************")

    segment_profiling_data = defaultdict(dict)

    closed_jaxpr, _ = trace_jaxpr_with_micro_batch(fun, [False] * len(avals), 1,
                                                   avals)
    unique_segment_mapping, forward_dot_counts = read_segment_mapping(
        "./debug_analysis_global/segment_mapping")
    picked_segments, unique_seg_deps = segment_jaxpr_by_pp_markers(
        closed_jaxpr, donated_invars, physical_mesh, logical_mesh_choices,
        as_option, unique_segment_mapping)

    start = time.time()
    compile_time = 0.0
    profile_time = 0.0
    segs_pb_specs = {}
    for segment in picked_segments:
        seg_idx, seg_name, seg_hlo, seg_final_closed_jaxpr, seg_donate_invars, seg_flop_count = segment
        outvar_id_list = unique_seg_deps.get_src_vars(seg_idx)
        invar_id_list = unique_seg_deps.get_dst_vars(seg_idx)
        avals = [v.aval for v in seg_final_closed_jaxpr.jaxpr.invars]
        seg_pb_specs = SegPbSpecs(seg_idx, invar_id_list, outvar_id_list, avals)
        # run auto sharding pass
        dump_profiles = int(os.getenv("DUMP_PROFILE_PARALLEL_BLOCK", 1))
        if (dump_profiles != 0):
            logger.info(f"RUN AutoSharding FOR SEGMENT {seg_idx} {seg_name}")
        seg_as_option = copy.deepcopy(as_option)
        seg_as_option.mode = "analysis_segment"
        seg_as_option.num_layers = 1
        seg_as_option.pb_config.seg_num = 1
        seg_as_option.pb_config.forward_dot_count = forward_dot_counts[seg_idx]
        seg_as_option.print_strategy = True

        seg_as_option.debug_dir = f'./debug_{seg_name}_{seg_as_option.mode}'
        run_as_exec(seg_final_closed_jaxpr,
                    seg_hlo,
                    logical_mesh_choices[0],
                    seg_as_option,
                    physical_mesh,
                    seg_donate_invars,
                    seg_flop_count,
                    sval=None,
                    skip_exec_test=True)

        #########################################################################

        seg_as_option.mode = "profile_segment"
        seg_as_option.print_strategy = False  # Set false when profiling multi pbs

        if (dump_profiles == 0):
            logger.info("Profiling...")
        reindex_list = read_reindex_items_list(seg_as_option.debug_dir +
                                               "/reindex_array")
        generate_search_space = check_reindex_list(reindex_list)
        if (dump_profiles != 0):
            logger.info(f"segment PB search space: {generate_search_space}")
        target_stras = enumerate_combinations(generate_search_space)
        overlap_compile_and_profiling = True
        if overlap_compile_and_profiling:
            overlapped_profiling(seg_idx,
                                 target_stras,
                                 segment_profiling_data,
                                 seg_final_closed_jaxpr,
                                 seg_hlo,
                                 logical_mesh_choices[0],
                                 seg_as_option,
                                 physical_mesh,
                                 seg_donate_invars,
                                 seg_flop_count,
                                 skip_exec_test=False,
                                 seg_pb_specs=seg_pb_specs,
                                 dump_profiles=dump_profiles)
        else:
            curr_optimal = float('inf')
            for pb_val in target_stras:
                c_start = time.time()
                exec = run_backend_compile(seg_final_closed_jaxpr,
                                           seg_hlo,
                                           logical_mesh_choices[0],
                                           seg_as_option,
                                           physical_mesh,
                                           seg_donate_invars,
                                           seg_flop_count,
                                           sval=pb_val,
                                           skip_exec_test=False,
                                           seg_pb_specs=seg_pb_specs)
                p_start = time.time()
                compile_time += (p_start - c_start)
                cost, mem_cost = run_exec(exec,
                                          pb_list=pb_val,
                                          mode=as_option.mode,
                                          curr_optimal=float('inf'))
                peak_memory = exec.get_total_allocation_size()
                profile_time += (time.time() - p_start)
                pb_val_ = tuple(pb_val)
                avg_cost = (cost[0] + cost[1] + cost[2]) / 3
                if (avg_cost < curr_optimal):
                    curr_optimal = avg_cost
                mem_cost = quantize_memory(mem_cost)
                peak_memory = quantize_memory(peak_memory)
                logger.info(
                    f"pb_val_ {pb_val} avgcost: {avg_cost} mem_cost(GB): {mem_cost} peak_memory(GB): {peak_memory}"
                )
                segment_profiling_data[seg_idx][pb_val_] = (avg_cost, mem_cost)
        segs_pb_specs[seg_idx] = seg_pb_specs
    if not overlap_compile_and_profiling:
        logger.info(f"s_compile time: {compile_time}")
        logger.info(f"s_profile time: {profile_time}")
    else:
        logger.info(f"overlapped_profiling overall time {time.time()-start}")

    # save segment_profiling_data, unique_seg_deps, segs_pb_specs
    segment_profiling_data = prune_profile_cost(segment_profiling_data)
    save_profiles(segment_profiling_data, unique_seg_deps, segs_pb_specs)
    exit()


def shard_parallel_internal_search(
        fun: lu.WrappedFun, in_tree: PyTreeDef, out_tree_thunk: Callable,
        static_argnums: Sequence[int], donated_invars: Sequence[bool],
        physical_mesh: PhysicalDeviceMesh,
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        as_option: AutoShardingOption, *avals: Sequence[AbstractValue]):
    assert as_option.mode == "search"
    logger.info("***********Search Pass****************")
    unique_segment_mapping, forward_dot_counts = read_segment_mapping(
        "./debug_analysis_global/segment_mapping")
    segment_profiling_data, unique_seg_deps, segs_pb_specs = load_profiles()
    pb_vals_lists, optimal_cost = build_space_and_search(
        unique_segment_mapping, segment_profiling_data, unique_seg_deps,
        segs_pb_specs, logical_mesh_choices[0])
    solution_list = pb_vals_lists  #+ m_pb_vals_lists
    # pylint: disable=unused-argument
    # Trace to get jaxpr
    closed_jaxpr, _ = trace_jaxpr_with_micro_batch(fun, [False] * len(avals), 1,
                                                   avals)
    name = f"{fun.__name__}_shard_parallel"
    init_hlo = jaxpr_to_hlo(name, closed_jaxpr, donated_invars)
    flop_count = xe.hlo_module_count_flop_dot_conv_only(init_hlo.get_module())
    as_option.mode = "search"
    as_option.print_strategy = True
    as_option.debug_dir = f'./debug_{as_option.mode}'
    as_option.pb_config.forward_dot_count = sum(forward_dot_counts)

    best_cost = float('inf')
    best_solution = None
    for solution in solution_list:
        exec, cost, mem_cost = run_as_exec(closed_jaxpr,
                                           init_hlo,
                                           logical_mesh_choices[0],
                                           as_option,
                                           physical_mesh,
                                           donated_invars,
                                           flop_count,
                                           sval=solution,
                                           force_profile_iters=1,
                                           skip_exec_test=False)
        avg_cost = np.mean(cost)
        if avg_cost < 0:
            continue
        if avg_cost < best_cost:
            best_solution = solution
            best_cost = avg_cost

    logger.info(f"searched ParallelBlock Strategies solution: {best_solution}")
    exec, cost, mem_cost = run_as_exec(closed_jaxpr,
                                       init_hlo,
                                       logical_mesh_choices[0],
                                       as_option,
                                       physical_mesh,
                                       donated_invars,
                                       flop_count,
                                       sval=best_solution,
                                       force_profile_iters=30,
                                       skip_exec_test=False)
    tflops = [(flop_count / 1e12) / c for c in cost]
    logger.info(f"TFLOP_COUNT: {flop_count/1e12}")
    logger.info(f"tflops: {tflops}")
    exit()
    return exec


def shard_parallel_internal_benchmark(
        fun: lu.WrappedFun, in_tree: PyTreeDef, out_tree_thunk: Callable,
        static_argnums: Sequence[int], donated_invars: Sequence[bool],
        physical_mesh: PhysicalDeviceMesh,
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        as_option: AutoShardingOption, *avals: Sequence[AbstractValue]):
    assert as_option.mode == "benchmark"
    os.environ["DUMP_PROFILE_PARALLEL_BLOCK"]="0"
    # pylint: disable=unused-argument
    # Trace to get jaxpr
    closed_jaxpr, _ = trace_jaxpr_with_micro_batch(fun, [False] * len(avals), 1,
                                                   avals)
    name = f"{fun.__name__}_shard_parallel"
    init_hlo = jaxpr_to_hlo(name, closed_jaxpr, donated_invars)
    flop_count = xe.hlo_module_count_flop_dot_conv_only(init_hlo.get_module())

    as_option.mode = "analysis_global"
    as_option.debug_dir = "./analysis_tmp"
    run_as_exec(closed_jaxpr,
                init_hlo,
                logical_mesh_choices[0],
                as_option,
                physical_mesh,
                donated_invars,
                flop_count,
                sval=None,
                skip_exec_test=True)

    reindex_list = read_reindex_items_list(as_option.debug_dir +
                                           "/reindex_array")
    generated_search_space = check_reindex_list(reindex_list)

    single_layer_pb_space_len = len(
        generated_search_space) // as_option.num_layers
    target_stras = []
    if as_option.pb_config.force_search_space:
        pbval_list = enumerate_combinations(
            as_option.pb_config.force_search_space)
    else:
        pbval_list = enumerate_combinations(
            generated_search_space[:single_layer_pb_space_len])

    for lis in pbval_list:
        target_stras.append(lis * as_option.num_layers)

    as_option.mode = "search"
    as_option.print_strategy = False
    os.environ["DUMP_COMM_VOLUME"] = "true"
    if as_option.pb_config.force_pb_lists:
        target_stras = as_option.pb_config.force_pb_lists
        os.environ["DUMP_COMM_VOLUME"] = "false"
    for sval in target_stras:
        exec, cost, mem_cost = run_as_exec(
            closed_jaxpr,
            init_hlo,
            logical_mesh_choices[0],
            as_option,
            physical_mesh,
            donated_invars,
            flop_count,
            sval=sval,
            skip_exec_test=False,
            force_profile_iters=2,
            dump_iters=1 if not as_option.pb_config.force_pb_lists else 0)
        print(f"sval {sval} {cost} {mem_cost}")
    os.environ["DUMP_COMM_VOLUME"] = "false"
    os.environ["DUMP_PROFILE_PARALLEL_BLOCK"]="1"
    exit()
    return exec
