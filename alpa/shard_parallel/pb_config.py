from dataclasses import dataclass, field
from typing import List, Optional


def default_str_field() -> Optional[str]:
    return field(default_factory=str)


@dataclass
class PBConfig:
    forward_dot_count: Optional[int] = 0
    seg_num: Optional[int] = 0
    force_search_space: Optional[List[int]] = field(default_factory=list)
    force_pb_lists: Optional[List[List[int]]] = field(default_factory=list)

    loaded_pb: Optional[str] = default_str_field()
    dot_pair_str: Optional[str] = default_str_field()
    merge_inst_str: Optional[str] = default_str_field()
    remove_dot_plan_str: Optional[str] = default_str_field()
    add_dot_plan_str: Optional[str] = default_str_field()
    new_dot_plan_str: Optional[str] = default_str_field()
    new_inst_plan_str: Optional[str] = default_str_field()
    force_search_space: Optional[str] = default_str_field()
    layers_to_segment: Optional[List[int]] = field(default_factory=lambda: [-1])
    segment_mapping: Optional[List[List[int]]] = field(default_factory=list)


default_pb_config = PBConfig()
