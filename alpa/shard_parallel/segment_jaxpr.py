from typing import Callable, Sequence, Optional, Union
import numpy as np
from jax.core import gensym

from jax.interpreters import pxla

from alpa.device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh
from alpa.global_env import global_config, CROSS_SEG_RESHARDING_BW
from alpa.shard_parallel.auto_sharding import AutoShardingOption

from alpa.pipeline_parallel.computation import (
    create_donation_mapping,
    generate_sharded_xla_computations_arguments_segment,
    get_donatable_intermediate, slice_closed_jaxpr_by_full_pipeline_marks,
    split_donate_invars)
from alpa.pipeline_parallel.compile_executable import split_and_process_layers, slice_apply_grad_for_stage_construction, shard_each_stage
from alpa.pipeline_parallel.stage_construction import (
    cluster_layers_and_slice_mesh, seperate_layers_as_segment, StageOption,
    UniformStageOption)

from alpa.pipeline_parallel.exp.exp_util import save_jaxpr

from collections import defaultdict
from dataclasses import dataclass
import time

dp_solutions = [[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 2]]


class SegPbSpecs:

    def __init__(self, seg_id, invar_list, outvar_list, avals):
        self.seg_id = seg_id
        self.invar_list = invar_list
        self.avals = avals
        self.outvar_list = outvar_list
        self.invars_pb_to_specs = {}
        self.outvars_pb_to_specs = {}

    def get_invar_list(self):
        return self.invar_list

    def get_outvar_list(self):
        return self.outvar_list

    def add_invars_pb_and_specs(self, pb_val, specs):
        pb_val = tuple(pb_val)
        self.invars_pb_to_specs[pb_val] = specs

    def add_outvars_pb_and_specs(self, pb_val, specs):
        pb_val = tuple(pb_val)
        self.outvars_pb_to_specs[pb_val] = specs

    def get_invars_specs(self, pb_val):
        pb_val = tuple(pb_val)
        return self.invars_pb_to_specs[pb_val]

    def get_outvars_specs(self, pb_val):
        pb_val = tuple(pb_val)
        return self.outvars_pb_to_specs[pb_val]

    def get_invar_specs(self, pb_val, invar_idx):
        pb_val = tuple(pb_val)
        invar_idx_ = self.invar_list.index(invar_idx)
        return self.invars_pb_to_specs[pb_val][invar_idx_]

    def get_outvar_specs(self, pb_val, outvar_idx):
        pb_val = tuple(pb_val)
        outvar_idx_ = self.outvar_list.index(outvar_idx)
        return self.outvars_pb_to_specs[pb_val][outvar_idx_]

    def get_aval(self, invar_idx):
        return self.avals[invar_idx]

    def dump(self):
        print(f"Segment ID: {self.seg_id}")


@dataclass
class SegmentDep:
    src_seg_id: int
    dst_seg_id: int
    src_var_id: int
    dst_var_id: int


class UniqueSegDeps:

    def __init__(self):
        self.all_deps = []
        self.seg_as_src_deps = defaultdict(list)
        self.seg_as_dst_deps = defaultdict(list)

    def add_seg_deps(self, src_seg_id, dst_seg_id, src_var_id, dst_var_id):
        seg_dep = SegmentDep(src_seg_id, dst_seg_id, src_var_id, dst_var_id)
        self.all_deps.append(seg_dep)

        self.seg_as_src_deps[src_seg_id].append(seg_dep)
        self.seg_as_dst_deps[dst_seg_id].append(seg_dep)

    def get_src_vars(self, seg_id):
        return list(
            {dep.src_var_id for dep in self.seg_as_src_deps.get(seg_id, [])})

    def get_dst_vars(self, seg_id):
        return list(
            {dep.dst_var_id for dep in self.seg_as_dst_deps.get(seg_id, [])})

    def get_var_pairs(self, src_seg_id, dst_seg_id):
        return [(dep.src_var_id, dep.dst_var_id)
                for dep in self.all_deps
                if dep.src_seg_id == src_seg_id and dep.dst_seg_id == dst_seg_id
               ]

    def dump(self):
        print("All Dependencies:")
        for dep in self.all_deps:
            print(
                f"  SrcSeg: {dep.src_seg_id}, DstSeg: {dep.dst_seg_id}, SrcVar: {dep.src_var_id}, DstVar: {dep.dst_var_id}"
            )
        print("\nDependencies by Source Segment:")
        for src, deps in self.seg_as_src_deps.items():
            print(f"  SrcSeg {src}:")
            for dep in deps:
                print(
                    f"    -> DstSeg: {dep.dst_seg_id}, SrcVar: {dep.src_var_id}, DstVar: {dep.dst_var_id}"
                )
        print("\nDependencies by Destination Segment:")
        for dst, deps in self.seg_as_dst_deps.items():
            print(f"  DstSeg {dst}:")
            for dep in deps:
                print(
                    f"    <- SrcSeg: {dep.src_seg_id}, SrcVar: {dep.src_var_id}, DstVar: {dep.dst_var_id}"
                )


def segment_dep_vars(src_seg, dst_seg):
    dep_vars = []
    src_indices = []
    dst_indices = []
    for i, var in enumerate(src_seg.jaxpr.outvars):
        if var in dst_seg.jaxpr.invars:
            dep_vars.append(var)
            src_indices.append(i)
            dst_indices.append(dst_seg.jaxpr.invars.index(var))
    return dep_vars, src_indices, dst_indices


def get_unique_seg_id(seg_id, unique_segment_mapping):
    unique_seg_id = -1
    for segs in unique_segment_mapping:
        if seg_id in segs:
            unique_seg_id = segs[0]
    return unique_seg_id


def check_segment_pair_consistency(segment_pair_dep_vars):
    """
    :param segment_pair_dep_vars: List of tuples (src_unique_seg_id, dst_unique_seg_id, dep_vars, src_indices, dst_indices)
    :return: A dictionary showing conflicts, or an empty dictionary if all are consistent.
    """
    pair_dict = {}
    unique_seg_deps = UniqueSegDeps()
    for item in segment_pair_dep_vars:
        src_unique_seg_id, dst_unique_seg_id, dep_vars, src_indices, dst_indices = item

        key = (src_unique_seg_id, dst_unique_seg_id)

        if key in pair_dict:
            if pair_dict[key] != (src_indices, dst_indices):
                print(f"Conflict found for {key}:")
                print(
                    f"Previous: {pair_dict[key]}, Current: ({src_indices}, {dst_indices})"
                )
                return None
        else:
            pair_dict[key] = (src_indices, dst_indices)
    for (src_unique_seg_id,
         dst_unique_seg_id), (src_indices, dst_indices) in pair_dict.items():
        for src_indice, dst_indice in zip(src_indices, dst_indices):
            unique_seg_deps.add_seg_deps(src_unique_seg_id, dst_unique_seg_id,
                                         src_indice, dst_indice)

    return unique_seg_deps


# return hlos for each picked segment
def segment_jaxpr_by_pp_markers(
        closed_jaxpr, donated_invars: Sequence[bool],
        physical_mesh: PhysicalDeviceMesh,
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        as_option: AutoShardingOption,
        unique_segment_mapping: Sequence[Sequence[int]]):

    save_jaxpr(closed_jaxpr, "##init_closed_jaxpr")
    #use functions in pipeline parallel to partition segments.
    global_invars = closed_jaxpr.jaxpr.invars
    gensym_func = gensym([closed_jaxpr.jaxpr])

    (closed_jaxpr, global_outvars, jax_pipeline_layers, apply_grad_jaxpr,
     microbatch_bound, reduction_vector, post_microbatch_bound,
     accumulator_mapping, acc_grad_invars, acc_grad_outvars
    ) = (
        split_and_process_layers(
            closed_jaxpr,
            full_batch_closed_jaxpr=None,
            num_microbatch=
            1,  # just split jaxpr to layers but not transform it to accumlation version.
            inference_mode=0,
            gensym_func=gensym_func))
    (jax_apply_layers,
     apply_grad_global_info) = slice_apply_grad_for_stage_construction(
         jax_pipeline_layers, apply_grad_jaxpr, microbatch_bound, global_invars,
         global_outvars, donated_invars, accumulator_mapping, gensym_func, 0)

    # save_jaxpr(jax_apply_layers, "##slice_apply_grad_for_stage_construction")

    # seperate layers, merge_marked_jaxpr_with_named_call will merge all non-pp_marker eqns and wrap it with new pp_marker,
    # we set wrap_with_markers=False to remove all pp markers.
    seperate_layers, stage_to_segment = seperate_layers_as_segment(
        jax_pipeline_layers, accumulator_mapping, acc_grad_outvars)
    donation_mapping = create_donation_mapping(accumulator_mapping,
                                               donated_invars, global_invars,
                                               global_outvars)
    donate_invars_dict, seperate_layers = split_donate_invars(
        donation_mapping, seperate_layers, gensym_func)
    num_segment = len(seperate_layers) // 2  # forward and backward
    print(f"NUM SEGMENT is {num_segment}")

    seg_dict = [[] for _ in range(num_segment)]
    seg_id_dict = [[] for _ in range(num_segment)]
    for i, layer in enumerate(seperate_layers):
        segment_idx = stage_to_segment[i]
        seg_dict[segment_idx].append(layer)
        seg_id_dict[segment_idx].append(i)

    segment_id_list = [segs[0] for segs in unique_segment_mapping]
    segments = []
    segment_jaxprs = []
    for seg_idx in range(num_segment):
        seg_donate_invars = [
            donate_invars_dict[layer_id] for layer_id in seg_id_dict[seg_idx]
        ]
        seg_name = f"segment_{seg_idx}"
        # merge forward and backward layers to one closed_jaxpr and transform it to hlo.
        # no pp marker here as generate_sharded_xla... just simply merge layers and do not remove pp marker.
        seg_hlo, final_closed_jaxpr, seg_flop_count = generate_sharded_xla_computations_arguments_segment(
            seg_name, seg_dict[seg_idx], seg_donate_invars)
        segment_jaxprs.append(final_closed_jaxpr)
        if seg_idx in segment_id_list:
            segments.append(
                tuple((seg_idx, seg_name, seg_hlo, final_closed_jaxpr,
                       seg_donate_invars, seg_flop_count)))
            save_jaxpr(final_closed_jaxpr, f"##segment_{seg_idx}")

    segment_pair_dep_vars = []
    for seg_pair_idx in range(num_segment - 1):
        dep_vars, src_indices, dst_indices = segment_dep_vars(
            segment_jaxprs[seg_pair_idx], segment_jaxprs[seg_pair_idx + 1])
        src_unique_seg_id = get_unique_seg_id(seg_pair_idx,
                                              unique_segment_mapping)
        dst_unique_seg_id = get_unique_seg_id(seg_pair_idx + 1,
                                              unique_segment_mapping)
        segment_pair_dep_vars.append((src_unique_seg_id, dst_unique_seg_id,
                                      dep_vars, src_indices, dst_indices))
    unique_seg_deps = check_segment_pair_consistency(segment_pair_dep_vars)
    # unique_seg_deps.dump()
    return segments, unique_seg_deps


def get_all_gather_cost(shape, dtype, src_spec, dst_spec, num_devices):
    latency = 0.0
    shard_mesh_dim = -1
    for i in range(len(shape)):
        sharding = src_spec.sharding[i]
        if isinstance(sharding, pxla.Chunked):
            for assignment in src_spec.mesh_mapping:
                if isinstance(assignment, pxla.ShardedAxis):
                    shard_mesh_dim = assignment.axis
                    break
    if (shard_mesh_dim > -1):
        if (dtype.name == 'float32'):
            byte_num = np.prod(shape) * 4
        else:
            byte_num = np.prod(shape) * 2
        cost = (num_devices - 1) / num_devices * byte_num
        latency = cost / CROSS_SEG_RESHARDING_BW
    return latency


def get_corss_seg_resharding_cost(
        cross_seg_id: int, first_pb: Sequence[int], second_pb: Sequence[int],
        unique_segment_mapping: Sequence[Sequence[int]],
        unique_seg_deps: UniqueSegDeps, seg_pb_specs: SegPbSpecs,
        logical_device_mesh: LogicalDeviceMesh):
    first_unique_seg_id = get_unique_seg_id(cross_seg_id,
                                            unique_segment_mapping)
    second_unique_seg_id = get_unique_seg_id(cross_seg_id + 1,
                                             unique_segment_mapping)
    dep_vars_pair = unique_seg_deps.get_var_pairs(first_unique_seg_id,
                                                  second_unique_seg_id)
    all_cost = 0.0
    for src_outvar_id, dst_invar_id in dep_vars_pair:
        aval = seg_pb_specs[second_unique_seg_id].get_aval(dst_invar_id)
        src_spec = seg_pb_specs[first_unique_seg_id].get_outvar_specs(
            first_pb, src_outvar_id)
        dst_spec = seg_pb_specs[second_unique_seg_id].get_invar_specs(
            second_pb, dst_invar_id)
        if src_spec == dst_spec:
            all_cost = all_cost + 0.0
        else:
            cost = get_all_gather_cost(aval.shape, aval.dtype, src_spec,
                                       dst_spec,
                                       logical_device_mesh.num_devices)
            all_cost = all_cost + cost
    return all_cost
    # print(f"unique_seg {first_unique_seg_id}: pb {first_pb} -> unique_seg {first_unique_seg_id}: pb {second_pb}.  all_cost {all_cost}")


def build_space_and_search(unique_segment_mapping, segment_profiling_data,
                           unique_seg_deps, segs_pb_specs,
                           logical_device_mesh: LogicalDeviceMesh):
    selected_pb_lists = []
    min_memory_pb_lists = []
    # build cost dict
    cost_dict = {}
    mem_cost_dict = {}
    pb_val_dicts = {}
    seg_num = sum(len(sublist) for sublist in unique_segment_mapping)

    for seg_idx in range(seg_num):
        unique_seg_id = get_unique_seg_id(seg_idx, unique_segment_mapping)
        sub_dict = {}
        sub_mem_dict = {}
        pb_val_dict = {}
        for stra_idx, (pb_val, (cost, mem_cost)) in enumerate(
                segment_profiling_data[unique_seg_id].items()):
            sub_dict[stra_idx] = cost
            sub_mem_dict[stra_idx] = mem_cost
            pb_val_dict[stra_idx] = pb_val
        cost_dict[seg_idx] = sub_dict
        mem_cost_dict[seg_idx] = sub_mem_dict
        pb_val_dicts[seg_idx] = pb_val_dict

    def get_cross_seg_cost(seg_id, stra_idx, success_stra_idx):
        return get_corss_seg_resharding_cost(
            seg_id, pb_val_dicts[seg_id][stra_idx],
            pb_val_dicts[seg_id + 1][success_stra_idx], unique_segment_mapping,
            unique_seg_deps, segs_pb_specs, logical_device_mesh)

    # add seg optimal to solution
    seg_optimal = [[] for _ in range(seg_num)]
    for segs in unique_segment_mapping:
        c_dict = cost_dict[segs[0]]
        min_cost_stra_idx = min(c_dict.items(), key=lambda x: x[1])[0]
        for seg_idx in segs:
            seg_optimal[seg_idx] = pb_val_dicts[segs[0]][min_cost_stra_idx]
    seg_solution = []
    for pb_vals in seg_optimal:
        for pb_val in pb_vals:
            seg_solution.append(pb_val)
    selected_pb_lists.append(seg_solution)

    cache = {}
    max_mem = global_config.MAX_MEM_SIZE * logical_device_mesh.num_devices
    start = time.time()

    def dp(seg_id, stra_id, used_mem):
        if seg_id == seg_num - 1:
            new_mem = used_mem + mem_cost_dict[seg_id][stra_id]
            if new_mem > max_mem:
                return float('inf'), []
            return cost_dict[seg_id][stra_id], [stra_id]

        if (seg_id, stra_id, used_mem) in cache:
            return cache[(seg_id, stra_id, used_mem)]

        cur_cost = cost_dict[seg_id][stra_id]
        cur_mem = mem_cost_dict[seg_id][stra_id]
        next_strategies = list(cost_dict[seg_id + 1].keys())

        best_result = None

        for next_stra_id in next_strategies:
            next_mem = mem_cost_dict[seg_id + 1][next_stra_id]
            total_mem = used_mem + cur_mem + next_mem

            if total_mem > max_mem:
                continue

            cross_cost = get_cross_seg_cost(seg_id, stra_id, next_stra_id)
            next_cost, next_path = dp(seg_id + 1, next_stra_id,
                                      used_mem + cur_mem)
            total_cost = cur_cost + cross_cost + next_cost
            current_path = [stra_id] + next_path

            if best_result is None or total_cost < best_result[0]:
                best_result = (total_cost, current_path)

        cache[(seg_id, stra_id,
               used_mem)] = best_result if best_result else (float('inf'), [])
        return cache[(seg_id, stra_id, used_mem)]

    best_overall = None
    for start_stra in cost_dict[0].keys():
        total_cost, path = dp(0, start_stra, 0)
        if best_overall is None or total_cost < best_overall[0]:
            best_overall = (total_cost, path)
    for solution in dp_solutions:
        selected_pb_lists.append(solution * seg_num)
    optimal_cost, optimal_path = best_overall
    acc_cost_pb_vals = []
    for seg_idx, stra_idx in enumerate(optimal_path):
        for pb_val in pb_val_dicts[seg_idx][stra_idx]:
            acc_cost_pb_vals.append(pb_val)
    selected_pb_lists.append(acc_cost_pb_vals)
    return selected_pb_lists, optimal_cost
