#include "tensorflow/compiler/xla/service/spmd/auto_sharding_opt_exp.h"

#include <algorithm>
#include <queue>
#include <regex>
#include <fstream>

#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

#include "tensorflow/compiler/xla/service/spmd/reindexing.pb.h"

namespace xla {
namespace spmd {

static bool CHECK_DEBUG_ENV(const std::string& name) {
  const char* value = std::getenv(name.c_str());
  if (value == nullptr) {
    return false;
  } else {
    std::string str(value);
    return (str == "1" || str == "true");
  }
}

static void PRINT_VECTOR(const std::string& prefix,
                         const std::vector<int>& vec) {
  std::cerr << prefix << ": ";
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cerr, " "));
  std::cerr << std::endl;
}

static void PRINT_VECTOR_IF_DEBUG(const std::string& env_var) {}
template <typename T, typename... Args>
static void PRINT_VECTOR_IF_DEBUG(const std::string& env_var,
                                  const std::string& prefix,
                                  const std::vector<T>& vec, Args&&... args) {
  if (CHECK_DEBUG_ENV(env_var)) {
    PRINT_VECTOR(prefix, vec);
    PRINT_VECTOR_IF_DEBUG(env_var, std::forward<Args>(args)...);
  } else {
  }
}

template <typename T>
std::string V2S(const std::vector<T>& vec) {
  std::stringstream ss;
  for (const auto& elem : vec) {
    ss << elem << ", ";
  }
  std::string result = ss.str();
  return result;
}

template <typename... Args>
static void PRINT_STRING_IF_DEBUG(const std::string& env_var,
                                  const std::string& prefix, Args&&... args) {
  if (CHECK_DEBUG_ENV(env_var)) {
    std::stringstream ss;
    ss << prefix;
    (ss << ... << std::forward<Args>(args));
    std::cerr << ss.str() << std::endl;
  }
}

void CheckEdgeCosts(const LeafStrategies& leaf_strategies, const CostGraph& cost_graph, std::vector<long int>& s_val){
  size_t N = leaf_strategies.size();
  for (size_t a = 0; a < N; ++a) {
    for (auto b : cost_graph.adjacency.at(a)) {
      if (a >= b) {
        continue;
      }
      double cost = cost_graph.edge_costs.at({a, b})(s_val[a], s_val[b]);
      if (cost > 0.9e+13) {
        LOG(ERROR) << "ERROR: " << a << " and " << b << " COST IS " << cost;
        throw std::runtime_error("ERROR: LARGE COST EDGE BETWEEN");
      }
    }
  }
}

// substract, multiply, add, divide, select, compare, exponet, rsqrt, maximum,
static bool check_op_kind(const HloInstruction* inst) {
  return (inst->opcode() == HloOpcode::kSubtract ||
          inst->opcode() == HloOpcode::kMultiply ||
          inst->opcode() == HloOpcode::kAdd ||
          inst->opcode() == HloOpcode::kDivide ||
          inst->opcode() == HloOpcode::kSelect ||
          inst->opcode() == HloOpcode::kCompare ||
          inst->opcode() == HloOpcode::kPower ||
          inst->opcode() == HloOpcode::kRsqrt ||
          inst->opcode() == HloOpcode::kSqrt ||
          inst->opcode() == HloOpcode::kPad ||
          inst->opcode() == HloOpcode::kExp ||
          inst->opcode() == HloOpcode::kSlice ||
          inst->opcode() == HloOpcode::kReduceWindow ||
          inst->opcode() == HloOpcode::kSelectAndScatter ||
          inst->opcode() == HloOpcode::kConvert ||
          inst->opcode() == HloOpcode::kMaximum ||
          inst->opcode() == HloOpcode::kDynamicUpdateSlice ||
          inst->opcode() == HloOpcode::kScatter ||
          inst->opcode() == HloOpcode::kConcatenate ||
          inst->opcode() == HloOpcode::kNegate);
}

// return <block_idx, inst_idx> if inst belongs to a parallel block.
std::pair<int, int> GetParallelBlock(const HloInstruction* inst,
                                     const std::vector<ParallelBlock>& blocks) {
  for (int i = 0; i < blocks.size(); ++i) {
    auto it = std::find(blocks[i].first.begin(), blocks[i].first.end(), inst);
    if (it != blocks[i].first.end()) {
      return std::make_pair(i, std::distance(blocks[i].first.begin(), it));
    }
  }
  return std::make_pair(-1, -1);
}

std::string DumpParallelBlocks(std::vector<ParallelBlock>& blocks,
                        const StrategyMap& strategy_map) {
  std::ostringstream os;
  for (int i = 0; i < blocks.size(); ++i) {
    os << "Parallel Block " << i << ":\n";
    for (int j = 0; j < blocks[i].first.size(); ++j) {
      const HloInstruction* inst = blocks[i].first[j];
      os << "    " << strategy_map.at(inst)->instruction_id << " "
                << inst->ToString(HloPrintOptions::ShortParsable()) << "\n";
      std::vector<int>& plan = blocks[i].second[j];
      // dump plan
      os << "    Plan: ";
      for (int k = 0; k < plan.size(); ++k) {
        os << plan[k] << ", ";
      }
      os << "\n";
    }
  }
  return os.str();
}

std::string DumpSVFile(const CostGraph& cost_graph, const StrategyMap& strategy_map) {
  std::ostringstream sv_file;
  for (int64_t i = 0; i < cost_graph.node_lens.size(); ++i) {
    sv_file << i << "," << cost_graph.follow_idx[i] << ","
            << cost_graph.node_lens[i];
    if (cost_graph.follow_idx[i] != -1) {
      for (auto j : cost_graph.reindexing_vector.at(i)) {
        sv_file << "," << j;
      }
    }
    sv_file << std::endl;
  }
  return sv_file.str();
}

void SaveReindexToFile(ReindexItemList& reindex_list, const std::string& file_name, bool append) {
  std::fstream output;
  if (append) {
      output.open(file_name, std::ios::out | std::ios::binary | std::ios::app);
  } else {
      output.open(file_name, std::ios::out | std::ios::binary | std::ios::trunc);
  }
  if (!output.is_open()){
    LOG(FATAL) << "Failed to open reindex file: " << file_name;
  }
  std::vector<ReindexItem> reindex_items(reindex_list.reindex_items().begin(), reindex_list.reindex_items().end());
  std::sort(reindex_items.begin(), reindex_items.end(), [](const ReindexItem& item1, const ReindexItem& item2) {
    return item1.inst_leaf_id() < item2.inst_leaf_id();
  });
  reindex_list.mutable_reindex_items()->Clear();
  for (const auto& item : reindex_items) {
    reindex_list.mutable_reindex_items()->Add()->CopyFrom(item);
  }
  reindex_list.SerializeToOstream(&output);
  output.close();
}

void LoadReindexFromFile(ReindexItemList& reindex_list, const std::string& file_path){
  std::string file_name = file_path + "/reindex_array";
  std::ifstream infile(file_name, std::ios::in | std::ios::binary);
  if (!infile.is_open()){
      LOG(FATAL) << "Failed to open reindex file: " << file_name;
  }
  reindex_list.ParseFromIstream(&infile);
}


void SaveReindexFile(const HloInstructionSequence& sequence, const CostGraph& cost_graph, 
    const LeafStrategies& leaf_strategies, const StrategyMap& strategy_map, 
    const std::string file_path, const std::vector<int>& failed_merge, const std::vector<ParallelBlock>& parallel_blocks, ReindexItemList& reindex_list){

  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  std::set<int> multi_pb_inst;
  // std::cerr << "multi_pb_inst \n "; 
  for (auto it : reindex_list.reindex_items()) {
        multi_pb_inst.insert(it.inst_id());
        // std::cerr << it.inst_id() << " ";
  }
  for (int64_t i = 0; i < cost_graph.node_lens.size(); ++i){
    
    bool in_failed_merge = false;
    for (auto it : failed_merge) {
      if (it == i) in_failed_merge = true;
    }
    ReindexItem item;
    int64_t target = cost_graph.follow_idx[i];
    item.set_inst_leaf_id(i);
    item.set_inst_id(leaf_strategies[i] -> instruction_id);
    const HloInstruction* inst = instructions[item.inst_id()];
    item.set_stra_num(cost_graph.node_lens[i]);
    bool key_inst = cost_graph.follow_idx[i] == -1 && (inst->opcode() == HloOpcode::kDot || inst->opcode() == HloOpcode::kConvolution);
    // DEBUG[20240929 huwf]: This make wrong index for reindexing list when there is index mismatch between inst id and leaf id, especially when adding pipeline markers in instructions.
    //bool deps_by_multi_pb = multi_pb_inst.count(i);
    bool deps_by_multi_pb = multi_pb_inst.count(leaf_strategies[i] -> instruction_id);
    if (deps_by_multi_pb && !key_inst) {
      //std::cerr << "[SaveReindex] Skip inst " << item.inst_id() << "\n";
      continue;
    }
    item.set_deps_by_multi_pb(deps_by_multi_pb);
    item.set_is_key_inst(key_inst);
    if (in_failed_merge && target == -1) {
        auto pb_id = GetParallelBlock(instructions[leaf_strategies[i]->instruction_id], parallel_blocks);
      	item.add_deps_pb_id(strategy_map.at(parallel_blocks[pb_id.first].first[0])->id);
        ReindexData* reindex_data = item.add_reindexing();
        std::vector<int> reindexing(parallel_blocks[pb_id.first].second[0].size());
        std::iota(reindexing.begin(), reindexing.end(), 0);
        for (auto r : reindexing){
          reindex_data -> add_reindex(r);
        }
        if (inst->opcode() == HloOpcode::kDot || inst->opcode() == HloOpcode::kConvolution) {
          item.set_is_key_inst(false);
        }
    }
    if ( target != -1){
        item.add_deps_pb_id(target);
        ReindexData* reindex_data = item.add_reindexing();
      	for (auto r : cost_graph.reindexing_vector.at(i)){
           reindex_data -> add_reindex(r);
        }
    }
    //std::cerr << "[SaveReindex] inst id : " << leaf_strategies[i] -> instruction_id << 
    //" leaf_id: " << i << " cost_graph.follow_idx is " << cost_graph.follow_idx[i]<<"\n";
    reindex_list.add_reindex_items()->CopyFrom(item);
  }
  std::string file_name = file_path + "/reindex_array";
  SaveReindexToFile(reindex_list, file_name, false);
}


void LoadParallelBlock(std::string loaded_pb,
                       const HloInstructionSequence& sequence,
                       ParallelBlock& parallel_block) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  std::string delimiter = ": ";
  size_t pos = loaded_pb.find(delimiter);
  std::string first_part = loaded_pb.substr(loaded_pb.find('{') + 1, pos - 2);
  std::string second_part = loaded_pb.substr(pos + delimiter.length());
  std::stringstream ss(first_part);
  std::string item;
  LOG(INFO) << "   Read Inst: " << first_part << "\n";
  while (std::getline(ss, item, ',')) {
    parallel_block.first.push_back(instructions[std::stoi(item)]);
  }
  std::stringstream ss2(second_part);
  std::string item2;
  std::vector<int> plan;
  LOG(INFO) << "   Read parallel plan: ";
  while (std::getline(ss2, item2, '[')) {
    if (item2.length() > 0) {
      LOG(INFO) << "[" << item2.substr(0, second_part.find(']')) << "], ";
      std::stringstream ss3(item2.substr(0, second_part.find(']')));
      std::string item3;
      plan.clear();
      while (std::getline(ss3, item3, ',')) {
        if (item3.length() > 0) {
          plan.push_back(std::stoi(item3));
        }
      }
      parallel_block.second.push_back(plan);
    }
  }
  LOG(INFO) << "\n";
}

void LoadParallelBlocks(std::string loaded_pbs,
                        const HloInstructionSequence& sequence,
                        std::vector<ParallelBlock>& parallel_blocks) {
  parallel_blocks.clear();
  std::stringstream ss(loaded_pbs);
  std::string loaded_pb;
  int idx = 0;
  while (std::getline(ss, loaded_pb, '%')) {
    LOG(INFO) << "\nRead Parallel Block " << idx++ << "\n";
    ParallelBlock parallel_block;
    LoadParallelBlock(loaded_pb, sequence, parallel_block);
    parallel_blocks.push_back(parallel_block);
  }
}

static bool is_reordered_shape(const Shape& A, const Shape& B) {
  if (A.dimensions().size() != B.dimensions().size()) {
    return false;
  }
  std::unordered_set<int64_t> setA(A.dimensions().begin(),
                                   A.dimensions().end());
  std::unordered_set<int64_t> setB(B.dimensions().begin(),
                                   B.dimensions().end());
  return setA == setB;
}

// return a user that is a dot instruction but not current_dot.
static HloInstruction* check_dot_user(
    HloInstruction* inst, const StrategyMap& strategy_map, int loss_pos,
    const absl::flat_hash_set<const HloInstruction*>& visited,
    std::vector<HloInstruction*>& path, const Shape& fw_dot_shape) {
  std::unordered_set<HloInstruction*> q_visited;
  std::queue<HloInstruction*> q;
  std::unordered_map<HloInstruction*, HloInstruction*> parent;
  q.push(inst);
  while (!q.empty()) {
    HloInstruction* cur = q.front();
    q.pop();
    for (HloInstruction* user : cur->users()) {
      if (q_visited.count(user) > 0) {
        continue;
      }
      q_visited.insert(user);
      parent[user] = cur;
      if ((user->opcode() == HloOpcode::kDot ||
           user->opcode() == HloOpcode::kConvolution) &&
          visited.count(user) == 0 &&
          strategy_map.at(user)->instruction_id > loss_pos) {
        const HloInstruction* bw_dot_input;
        // check if the input shape of current dot match forward dot output shape.
        if (is_reordered_shape(user->operand(0)->shape(), fw_dot_shape) || is_reordered_shape(user->operand(1)->shape(), fw_dot_shape)) {
          while (user != inst) {
            path.push_back(user);
            user = parent[user];
          }
          path.push_back(inst);
          std::reverse(path.begin(), path.end());
          return path.back();
        }else{
          LOG(INFO) << "Shared input but not backward dot, continue search dot user.\n";
        }
      } else if (user->opcode() == HloOpcode::kReshape ||
                 user->opcode() == HloOpcode::kTranspose ||
                 user->opcode() == HloOpcode::kReverse) {
        q.push(user);
      }
    }
  }
  return nullptr;
}

static HloInstruction* find_backward_dot(
    const HloInstructionSequence& sequence, const StrategyMap& strategy_map,
    int loss_pos, HloInstruction* inst,
    const absl::flat_hash_set<const HloInstruction*>& visited,
    const Shape& fw_dot_shape, std::vector<HloInstruction*>& to_shared_operand,
    std::vector<HloInstruction*>& from_shared_operand) {
  std::vector<HloInstruction*> shared_operands;
  // use a map to record each instruction's parent.
  std::unordered_map<HloInstruction*, HloInstruction*> parent;
  HloInstruction* ins = inst;
  shared_operands.push_back(ins);
  while (ins->opcode() == HloOpcode::kReshape ||
         ins->opcode() == HloOpcode::kTranspose ||
         ins->opcode() == HloOpcode::kReverse) {
    shared_operands.push_back(ins->operands()[0]);
    parent[ins->operands()[0]] = ins;
    ins = ins->operands()[0];
  }
  
  for (HloInstruction* op : shared_operands) {
    HloInstruction* dot_user = check_dot_user(op, strategy_map, loss_pos,
                                              visited, from_shared_operand, fw_dot_shape);

    if (dot_user != nullptr) {
      const HloInstruction* bw_dot_input;
      // check if the input shape of current dot match forward dot output shape.
      if (dot_user->operand(0) ==
          from_shared_operand[from_shared_operand.size() - 2]) {
        bw_dot_input = dot_user->operand(1);
      } else {
        bw_dot_input = dot_user->operand(0);
      }

      if (!is_reordered_shape(bw_dot_input->shape(), fw_dot_shape)) {
        LOG(INFO) << "Shared input but not backward dot.\n";
        continue;
      }
      // rebuild the path from op to inst.
      while (op != inst) {
        to_shared_operand.push_back(op);
        op = parent[op];
      }
      to_shared_operand.push_back(inst);
      std::reverse(to_shared_operand.begin(), to_shared_operand.end());
      return dot_user;
    }
  }
  return nullptr;
}

// only consider these cases:
// * Adding or removing dimensions with size 1.
// * Merging consecutive dimensions.
// * Splitting a dimension to consecutive dimensions.
// * [LLAMA FIXME]
// * FROM LLAMA2. size of source_shape - size of target_shape == 2
static void deps_for_reshape(const Shape& source_shape,
                             const Shape& target_shape,
                             std::vector<int>& map_vector) {
  map_vector.clear();
  if (source_shape.rank() == target_shape.rank()) {
    // std::cerr << "WARRNING: reshape with same rank, can't handle this.\n";
    for (int i = 0; i < source_shape.rank(); ++i) {
      map_vector.push_back(i);
    }
    return;
  }
  
  auto isPrefixedWithOne = [](const Shape& source_shape, const Shape& target_shape) {
    if (source_shape.rank() != target_shape.rank() - 1) {
        return false;   
    }
    if (target_shape.dimensions(0) != 1) {
        return false; 
    }
    for (size_t i = 0; i < source_shape.rank(); ++i) {
        if (source_shape.dimensions(i) != target_shape.dimensions(i+1)) {
            return false; 
        }
    }
    return true; 
  };
  if (isPrefixedWithOne(source_shape, target_shape)) {
    map_vector.push_back(-1);
    for (int i = 1; i<target_shape.rank(); ++i){
      map_vector.push_back(i-1);
    }
    return; 
  }

  std::vector<int64_t> source_dims_stack(source_shape.rank());
  std::vector<int64_t> target_dims_stack(target_shape.rank());
  for (int64_t i = 0; i < source_shape.rank(); ++i) {
    source_dims_stack[i] = source_shape.dimensions(i);
  }
  for (int64_t i = 0; i < target_shape.rank(); ++i) {
    target_dims_stack[i] = target_shape.dimensions(i);
  }
   // fuck strange reshape in llama 70b
   // [1,8,8,128,8192] -> [8192, 8192]
  if (source_shape.rank() - target_shape.rank() == 3) {
    if (source_dims_stack[0] == 1){
      map_vector.push_back(1);
    } else {
      map_vector.push_back(0);
    }
      map_vector.push_back(4);
    // std::cerr << "[WARRNING][WARRNING][WARRNING][WARRNING] \n";
    // std::cerr << " Source: " ;
    // for (auto i : source_dims_stack) { std::cerr << i << " ";}
    // std::cerr << "\n";
    // std::cerr << " Target: " ;
    // for (auto i : target_dims_stack) { std::cerr << i << " ";}
    // std::cerr << "\n";
    return;
  } 
  int redundant_dim_pos = 0;
  if (source_shape.rank() - target_shape.rank() == 2){
    auto find_in_source = std::find(source_dims_stack.begin(), source_dims_stack.end(), 1);
    redundant_dim_pos = std::distance(source_dims_stack.begin(), find_in_source);
    source_dims_stack.erase(std::remove_if(source_dims_stack.begin(), source_dims_stack.end(), [](int value) {
        return value == 1;
    }),source_dims_stack.end());
  }
  // [128, 128, 256] -> [128, 128, 8, 1, 32] in LLAMA
  if (target_shape.rank() - source_shape.rank() == 2) {
    auto find_in_target = std::find(target_dims_stack.begin(), target_dims_stack.end(), 1);
    redundant_dim_pos = std::distance(target_dims_stack.begin(), find_in_target);
    target_dims_stack.erase(std::remove_if(target_dims_stack.begin(), target_dims_stack.end(), [](int value) {
        return value == 1;
    }),target_dims_stack.end());
  }
  // [8192 8 8 128] -> [8192, 8192]
  if (source_dims_stack.size() - target_dims_stack.size() == 2) {
    // still have two more dimension after erase 1.
    map_vector.push_back(0);
    map_vector.push_back(3);
    return;
  }

  // Normal reshape
  while (!source_dims_stack.empty() && !target_dims_stack.empty()) {
    if (target_dims_stack.empty()) {
      break;
    }
    int64_t s_size = 1;
    int64_t t_size = 1;
    if (!source_dims_stack.empty()) {
      s_size = source_dims_stack.back();
      source_dims_stack.pop_back();
    }
    t_size = target_dims_stack.back();
    target_dims_stack.pop_back();
    // same dimension
    if (s_size == t_size) {
      map_vector.push_back(source_dims_stack.size());
    }
    // add a new dimension in dst
    else if (t_size == 1) {
      map_vector.push_back(-1);
      source_dims_stack.push_back(s_size);
    }
    // remove a dimension in dst
    else if (s_size == 1) {
      // skip this dimension
      target_dims_stack.push_back(t_size);
    } else if (s_size > t_size) {
      map_vector.push_back(source_dims_stack.size());
      source_dims_stack.push_back(s_size / t_size);

    } else {
      // skip this dimension
      CHECK(!source_dims_stack.empty());
      source_dims_stack.back() *= s_size;
      target_dims_stack.push_back(t_size);
      CHECK_EQ(source_dims_stack.back(), t_size);
    }
  }

  std::reverse(map_vector.begin(), map_vector.end());
  // adjust map_vector if we have remove a dimension of source_dim_stack
  if (source_shape.rank() - target_shape.rank() == 2) {
    for (int pos=0; pos < map_vector.size(); ++pos){
      if (pos >= redundant_dim_pos){
        map_vector[pos] += 1;
      }
    }
  }
  // insert -1 to map_vector if we have remove a dimension of target_dim_stack
  if (target_shape.rank() - source_shape.rank() == 2) {
    map_vector.insert(map_vector.begin() + redundant_dim_pos, -1);
  }

  std::vector<int> tmp(map_vector.size());
  if (source_shape.rank() == 1 && target_shape.rank() > 2) {
    for (int idx = 0; idx < target_shape.rank(); ++idx) {
      if (target_shape.dimensions(idx) != 1 &&
          target_shape.dimensions(idx) != source_shape.dimensions(0)) {
        return;
      }
      if (target_shape.dimensions(idx) == 1) {
        tmp[idx] = -1;
      } else {
        tmp[idx] = 0;
      }
    }
    map_vector = tmp;
  }
}

static void dim_dep_from_path_for_bw(std::vector<HloInstruction*>& path,
                                     std::vector<int>& dim_dep) {
  // build dimension level dependence from src to dst in a simple way, only
  // consider reshape/transpose/reverse and elem-wise ops beacuse others are not
  // likely to appear in the dependency path between forward/backward dot and
  // their commen user.

  // shared_op -> xxx -> dot_x input
  std::iota(dim_dep.begin(), dim_dep.end(), 0);
  path.erase(path.begin());
  PRINT_VECTOR_IF_DEBUG("DEBUG_DEPS", "start deps: ", dim_dep);
  for (auto inst : path) {
    // std::cerr << inst->ToString(HloPrintOptions::ShortParsable()) << "\n";
    int64_t current_dim_size = inst->shape().rank();
    assert(inst->shape().dimensions_size() == dim_dep.size());
    std::vector<int> map_vector(current_dim_size);
    if (inst->opcode() == HloOpcode::kTranspose) {
      for (int64_t i = 0; i < inst->dimensions().size(); ++i) {
        map_vector[i] = inst->dimensions()[i];
      }
    } else if (inst->opcode() == HloOpcode::kReverse) {
      std::iota(map_vector.begin(), map_vector.end(), 0);
      std::swap(map_vector[inst->dimensions()[0]],
                map_vector[inst->dimensions()[1]]);
    } else if (inst->opcode() == HloOpcode::kReshape) {
      map_vector.clear();
      deps_for_reshape(inst->operands()[0]->shape(), inst->shape(), map_vector);
    } else {
      std::iota(map_vector.begin(), map_vector.end(), 0);
    }
    std::vector<int> new_dim_dep(inst->shape().rank());
    for (int64_t i = 0; i < current_dim_size; ++i) {
      if (map_vector[i] < 0) {
        new_dim_dep[i] = -1;
      } else {
        new_dim_dep[i] = (dim_dep[map_vector[i]]);
      }
    }
    dim_dep = new_dim_dep;
    PRINT_VECTOR_IF_DEBUG("DEBUG_DEPS", "   map_vector: ", map_vector,
                          "mapped_deps", dim_dep);
  }
}

static std::vector<int> infer_dim_dep_from_shared_op(
    const std::vector<int>& fvec, const std::vector<int>& bvec) {
  int skipped = 0;
  int skipped_index = -1;
  int skipped_index_in_b = -1;
  std::vector<int> f_vec = fvec;
  std::vector<int> b_vec = bvec;
  for (int i = 0; i < b_vec.size(); ++i) {
    auto it = find(f_vec.begin(), f_vec.end(), b_vec[i]);
    int index = it - f_vec.begin();
    if (it == f_vec.end()) {
      skipped++;
      skipped_index = i;
      skipped_index_in_b = i;
      continue;
    }
    f_vec[index] = INT_MAX;
    b_vec[i] = index;
  }

  if (skipped > 1) {
    LOG(FATAL) << "[ERROR]: Cant Handle This Dependence Pattern.\n";
  } else if (skipped == 1) {
    auto it = find_if(f_vec.begin(), f_vec.end(),
                      [skipped_index](int x) { return (x != INT_MAX); });
    if (it != f_vec.end()) {
      int index = it - f_vec.begin();
      // b_vec[skipped_index] = std::min(b_vec[skipped_index], f_vec[index]);
      b_vec[skipped_index] = index;
    } else {
      LOG(FATAL) << "[ERROR]: Cant Handle This Dependence Pattern..\n";
    }
  }
  return b_vec;
}

static int RemainDimIndexForDotOp(
    int dim_number, const tsl::protobuf::RepeatedField<int64_t>& batch_dims,
    const tsl::protobuf::RepeatedField<int64_t>& contract_dims) {
  std::vector<int> a(dim_number);
  generate(a.begin(), a.end(), [n = 0]() mutable { return n++; });

  auto end =
      remove_if(a.begin(), a.end(), [&batch_dims, &contract_dims](int x) {
        return std::find(batch_dims.begin(), batch_dims.end(), x) !=
                   batch_dims.end() ||
               std::find(contract_dims.begin(), contract_dims.end(), x) !=
                   contract_dims.end();
      });
  a.erase(end, a.end());
  if (a.size() == 1) {
    return a[0];
  } else {
    return -1;
  }
}

static std::pair<HloSharding, HloSharding> get_operand_sharding_from_conv(
    const HloInstruction* conv, const std::string& stra_name,
    const ClusterEnvironment& cluster_env, std::vector<int>& lhs_dev_dim,
    std::vector<int>& rhs_dev_dim, std::vector<int>& out_dev_dim) {
  if (out_dev_dim.empty()) {
    out_dev_dim = std::vector<int>(conv->shape().rank(), -1);
  }
  const HloInstruction* lhs = conv->operand(0);
  const HloInstruction* rhs = conv->operand(1);
  const Array<int64_t>& device_mesh = cluster_env.device_mesh;
  const Array<int64_t>& device_mesh_1d = cluster_env.device_mesh_1d;
  const ConvolutionDimensionNumbers& conv_dnums =
      conv->convolution_dimension_numbers();
  int64_t lhs_batch_dim = conv_dnums.input_batch_dimension();
  int64_t lhs_in_channel_dim = conv_dnums.input_feature_dimension();
  int64_t rhs_in_channel_dim = conv_dnums.kernel_input_feature_dimension();
  int64_t rhs_out_channel_dim = conv_dnums.kernel_output_feature_dimension();
  int64_t out_batch_dim = conv_dnums.output_batch_dimension();
  int64_t out_out_channel_dim = conv_dnums.output_feature_dimension();
  std::regex pattern_0(
      "SS = SR x RS @ \\{(\\d+),(\\d+)\\}");  // SplitLhsBatchRhsOutChannel
  std::regex pattern_1(
      "SR = SS x SR @ \\{(\\d+),(\\d+)\\} \\(allreduce @ (\\d+)\\)");  // SplitLhsBatchBothInchannel
  std::regex pattern_2(
      "RS = RS x SS @ \\{(\\d+),(\\d+)\\} \\(allreduce @ (\\d+)\\)");  // SplitRhsOutchannelBothInchannel
  std::regex pattern_3("Si = Si x R @ 0");  // Add1DDataParallel Si = Si x R @ 0
  std::regex pattern_4(
      "R = Sk x Sk @ (\\d+) \\(allreduce @ (\\d+)\\)");  // Add1DDataParallel R
                                                         // = Sk x Sk @ 0
                                                         // (allreduce @ 0)
  // SS = SS x RS @ {%d,%d}
  std::regex pattern_5(
      "SS = SS x RS @ \\{(\\d+),(\\d+)\\}");  // SplitDepthwise need figure out
                                              // different between forward and
                                              // backward
  std::vector<std::regex> patterns = {pattern_0, pattern_1, pattern_2,
                                      pattern_3, pattern_4, pattern_5};
  bool matched = false;
  for (int i = 0; i < patterns.size(); ++i) {
    std::smatch match;
    if (std::regex_match(stra_name, match, patterns[i])) {
      std::vector<int> extract_var;
      for (int j = 1; j < match.size(); ++j) {
        extract_var.push_back(std::stoi(match[j].str()));
      }
      int mesh_dim0, mesh_dim1;
      if (i == 0) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dim}, {mesh_dim0}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_out_channel_dim}, {mesh_dim1}, device_mesh);
        lhs_dev_dim[lhs_batch_dim] = mesh_dim0;
        rhs_dev_dim[rhs_out_channel_dim] = mesh_dim1;
        out_dev_dim[out_batch_dim] = mesh_dim0;
        out_dev_dim[out_out_channel_dim] = mesh_dim1;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 1) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dim, lhs_in_channel_dim},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_in_channel_dim}, {mesh_dim1}, device_mesh);
        lhs_dev_dim[lhs_batch_dim] = mesh_dim0;
        lhs_dev_dim[lhs_in_channel_dim] = mesh_dim1;
        rhs_dev_dim[rhs_in_channel_dim] = mesh_dim1;
        out_dev_dim[out_batch_dim] = mesh_dim0;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 2) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_in_channel_dim}, {mesh_dim0}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_in_channel_dim, rhs_out_channel_dim},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        lhs_dev_dim[lhs_in_channel_dim] = mesh_dim0;
        rhs_dev_dim[rhs_in_channel_dim] = mesh_dim0;
        rhs_dev_dim[rhs_out_channel_dim] = mesh_dim1;
        out_dev_dim[out_out_channel_dim] = mesh_dim1;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 3) {
        int mesh_dim = 0;
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dim}, {mesh_dim}, device_mesh_1d);
        HloSharding rhs_spec = HloSharding::Replicate();
        lhs_dev_dim[lhs_batch_dim] = mesh_dim;
        out_dev_dim[out_batch_dim] = mesh_dim;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 4) {
        int mesh_dim = 0;
        HloSharding lhs_spec = Tile(lhs->shape(), {lhs_in_channel_dim},
                                    {mesh_dim}, device_mesh_1d);
        HloSharding rhs_spec = Tile(rhs->shape(), {rhs_in_channel_dim},
                                    {mesh_dim}, device_mesh_1d);
        lhs_dev_dim[lhs_in_channel_dim] = mesh_dim;
        rhs_dev_dim[rhs_in_channel_dim] = mesh_dim;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 5) {
        // need special handle for difference of forward and backward.
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        if ((conv->feature_group_count() ==
                 lhs->shape().dimensions(lhs_in_channel_dim) &&
             conv->feature_group_count() ==
                 rhs->shape().dimensions(rhs_out_channel_dim))) {
          // forward == True
          HloSharding lhs_spec =
              Tile(lhs->shape(), {lhs_batch_dim, lhs_in_channel_dim},
                   {mesh_dim0, mesh_dim1}, device_mesh);
          HloSharding rhs_spec = Tile(rhs->shape(), {rhs_out_channel_dim},
                                      {mesh_dim1}, device_mesh);
          lhs_dev_dim[lhs_batch_dim] = mesh_dim0;
          lhs_dev_dim[lhs_in_channel_dim] = mesh_dim1;
          rhs_dev_dim[rhs_out_channel_dim] = mesh_dim1;
          out_dev_dim[out_batch_dim] = mesh_dim0;
          out_dev_dim[out_out_channel_dim] = mesh_dim1;
          return std::make_pair(lhs_spec, rhs_spec);

        } else if ((conv->batch_group_count() ==
                        lhs->shape().dimensions(lhs_batch_dim) &&
                    conv->batch_group_count() ==
                        rhs->shape().dimensions(rhs_out_channel_dim))) {
          // forward == False
          HloSharding lhs_spec =
              Tile(lhs->shape(), {lhs_batch_dim, lhs_in_channel_dim},
                   {mesh_dim1, mesh_dim0}, device_mesh);
          HloSharding rhs_spec = Tile(rhs->shape(), {rhs_out_channel_dim},
                                      {mesh_dim1}, device_mesh);
          lhs_dev_dim[lhs_batch_dim] = mesh_dim1;
          lhs_dev_dim[lhs_in_channel_dim] = mesh_dim0;
          rhs_dev_dim[rhs_out_channel_dim] = mesh_dim1;
          out_dev_dim[out_batch_dim] = mesh_dim1;
          out_dev_dim[out_out_channel_dim] = mesh_dim1;
          return std::make_pair(lhs_spec, rhs_spec);
        }
      }
      matched = true;
      break;
    }
  }
  if (!matched) {
    std::cerr << "[ERROR]: can not parse the sharding string: " << stra_name
              << "\n";
    HloSharding lhs_spec = Tile(lhs->shape(), {0}, {0}, device_mesh);
    HloSharding rhs_spec = Tile(rhs->shape(), {0}, {0}, device_mesh);
    return std::make_pair(lhs_spec, rhs_spec);
  }
}

static std::pair<HloSharding, HloSharding> get_operand_sharding_from_dot(
    const HloInstruction* dot, const std::string& stra_name,
    const ClusterEnvironment& cluster_env, std::vector<int>& lhs_dev_dim,
    std::vector<int>& rhs_dev_dim, std::vector<int>& out_dev_dim) {
  if (out_dev_dim.empty()) {
    out_dev_dim = std::vector<int>(dot->shape().rank(), -1);
  }
  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  const Array<int64_t>& device_mesh = cluster_env.device_mesh;
  const Array<int64_t>& device_mesh_1d = cluster_env.device_mesh_1d;
  const DotDimensionNumbers& dot_dnums = dot->dot_dimension_numbers();
  const tsl::protobuf::RepeatedField<int64_t>& lhs_con_dims =
      dot_dnums.lhs_contracting_dimensions();
  const tsl::protobuf::RepeatedField<int64_t>& rhs_con_dims =
      dot_dnums.rhs_contracting_dimensions();
  const tsl::protobuf::RepeatedField<int64_t>& lhs_batch_dims =
      dot_dnums.lhs_batch_dimensions();
  const tsl::protobuf::RepeatedField<int64_t>& rhs_batch_dims =
      dot_dnums.rhs_batch_dimensions();
  std::vector<int64_t> lhs_space_dims, rhs_space_dims;
  std::pair<std::vector<int64_t>, std::vector<int64_t>> lhs_rhs_space_dims =
      GetSpaceDims(lhs->shape(), rhs->shape(), dot_dnums);
  lhs_space_dims = lhs_rhs_space_dims.first;
  rhs_space_dims = lhs_rhs_space_dims.second;
  int64_t out_lhs_space_dim = dot_dnums.lhs_batch_dimensions_size();
  int64_t out_rhs_space_dim = out_lhs_space_dim + 1;

  std::regex pattern_0(
      "SS = SR x RS @ \\{(\\d+),(\\d+)\\}");  // SplitLhsSpaceRhsSpace
  std::regex pattern_1(
      "SR = SS x SR @ \\{(\\d+),(\\d+)\\} \\(allreduce @ (\\d+)\\)");  // SplitLhsSpaceBothContract
  std::regex pattern_2(
      "RS = RS x SS @ \\{(\\d+),(\\d+)\\} \\(allreduce @ (\\d+)\\)");  // SplitRhsSpaceBothContract
  std::regex pattern_3(
      "Sb_(\\d+) = Sb x Sb @ \\{(\\d+)\\}");  // SplitOneBatchDim
  std::regex pattern_4(
      "Sb = Sb x Sb @ \\{(\\d+),(\\d+)\\}");  // SplitTwoBatchDims
  std::regex pattern_5(
      "SbSi = SbSi x SbR @ \\{(\\d+),(\\d+)\\}");  // SplitBatchDimLhsSpace
  std::regex pattern_6(
      "SbSj = SbR x SbSj @ \\{(\\d+),(\\d+)\\}");  // SplitBatchDimRhsSpace
  std::regex pattern_7(
      R"(SbR = SbSk x SbSk @ \{(\d+),(\d+)\} )"
      R"(\(allreduce @ (\d+)\})");  // SplitBatchDimBothContract
  std::regex pattern_8(
      "RR = RS x SR @ \\{(\\d+)\\} \\(allreduce @ (\\d+)\\)");  // RecomputeSplitBothContract
  std::regex pattern_9("Si = Si x R @ (\\d+)");  // Add1DDataParallel (2D case)
  std::regex pattern_10(
      "R = Sk x Sk @ (\\d+) \\(allreduce @ (\\d+)\\)");  // Add1DDataParallel
                                                         // (2D case)
  std::regex pattern_11(
      "Sb_(\\d+) = Sb x Sb @ \\{(\\d+)\\} 1d");  // Add1DBatchSplit (2D case)
  std::vector<std::regex> patterns = {
      pattern_0, pattern_1, pattern_2, pattern_3, pattern_4,  pattern_5,
      pattern_6, pattern_7, pattern_8, pattern_9, pattern_10, pattern_11};
  bool matched = false;
  for (int i = 0; i < patterns.size(); ++i) {
    std::smatch match;
    if (std::regex_match(stra_name, match, patterns[i])) {
      std::vector<int> extract_var;
      for (int j = 1; j < match.size(); ++j) {
        extract_var.push_back(std::stoi(match[j].str()));
      }
      int mesh_dim0, mesh_dim1;
      if (i == 0) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_space_dims[0]}, {mesh_dim0}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_space_dims[0]}, {mesh_dim1}, device_mesh);
        lhs_dev_dim[lhs_space_dims[0]] = mesh_dim0;
        rhs_dev_dim[rhs_space_dims[0]] = mesh_dim1;
        out_dev_dim[out_lhs_space_dim] = mesh_dim0;
        out_dev_dim[out_rhs_space_dim] = mesh_dim1;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 1) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_space_dims[0], lhs_con_dims[0]},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_con_dims[0]}, {mesh_dim1}, device_mesh);
        lhs_dev_dim[lhs_space_dims[0]] = mesh_dim0;
        lhs_dev_dim[lhs_con_dims[0]] = mesh_dim1;
        rhs_dev_dim[rhs_con_dims[0]] = mesh_dim1;
        out_dev_dim[out_lhs_space_dim] = mesh_dim0;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 2) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_con_dims[0]}, {mesh_dim0}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_con_dims[0], rhs_space_dims[0]},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        lhs_dev_dim[lhs_con_dims[0]] = mesh_dim0;
        rhs_dev_dim[rhs_con_dims[0]] = mesh_dim0;
        rhs_dev_dim[rhs_space_dims[0]] = mesh_dim1;
        out_dev_dim[out_rhs_space_dim] = mesh_dim1;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 3) {
        int dim = extract_var[0];
        int num = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dims[dim]}, {num}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_batch_dims[dim]}, {num}, device_mesh);
        lhs_dev_dim[lhs_batch_dims[dim]] = num;
        rhs_dev_dim[rhs_batch_dims[dim]] = num;
        out_dev_dim[lhs_batch_dims[dim]] = num;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 4) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dims[0], lhs_batch_dims[1]},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_batch_dims[0], rhs_batch_dims[1]},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        lhs_dev_dim[lhs_batch_dims[0]] = mesh_dim0;
        lhs_dev_dim[lhs_batch_dims[1]] = mesh_dim1;
        rhs_dev_dim[rhs_batch_dims[0]] = mesh_dim0;
        rhs_dev_dim[rhs_batch_dims[1]] = mesh_dim1;
        out_dev_dim[lhs_batch_dims[0]] = mesh_dim0;
        out_dev_dim[lhs_batch_dims[1]] = mesh_dim1;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 5) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dims[0], lhs_space_dims[0]},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_batch_dims[0]}, {mesh_dim0}, device_mesh);
        lhs_dev_dim[lhs_batch_dims[0]] = mesh_dim0;
        lhs_dev_dim[lhs_space_dims[0]] = mesh_dim1;
        rhs_dev_dim[rhs_batch_dims[0]] = mesh_dim0;
        out_dev_dim[lhs_batch_dims[0]] = mesh_dim0;
        out_dev_dim[out_lhs_space_dim] = mesh_dim1;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 6) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dims[0]}, {mesh_dim0}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_batch_dims[0], rhs_space_dims[0]},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        lhs_dev_dim[lhs_batch_dims[0]] = mesh_dim0;
        rhs_dev_dim[rhs_batch_dims[0]] = mesh_dim0;
        rhs_dev_dim[rhs_space_dims[0]] = mesh_dim1;
        out_dev_dim[lhs_batch_dims[0]] = mesh_dim0;
        out_dev_dim[out_rhs_space_dim] = mesh_dim1;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 7) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dims[0], lhs_con_dims[0]},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_batch_dims[0], rhs_con_dims[0]},
                 {mesh_dim0, mesh_dim1}, device_mesh);
        lhs_dev_dim[lhs_batch_dims[0]] = mesh_dim0;
        lhs_dev_dim[lhs_con_dims[0]] = mesh_dim1;
        rhs_dev_dim[rhs_batch_dims[0]] = mesh_dim0;
        rhs_dev_dim[rhs_con_dims[0]] = mesh_dim1;
        out_dev_dim[lhs_batch_dims[0]] = mesh_dim0;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 8) {
        mesh_dim0 = extract_var[0];
        mesh_dim1 = extract_var[1];
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_con_dims[0]}, {mesh_dim0}, device_mesh);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_con_dims[0]}, {mesh_dim0}, device_mesh);
        lhs_dev_dim[lhs_con_dims[0]] = mesh_dim0;
        rhs_dev_dim[rhs_con_dims[0]] = mesh_dim0;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 9) {
        int mesh_dim = 0;
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_space_dims[0]}, {mesh_dim}, device_mesh_1d);
        HloSharding rhs_spec = HloSharding::Replicate();
        lhs_dev_dim[lhs_space_dims[0]] = mesh_dim;
        out_dev_dim[out_lhs_space_dim] = mesh_dim;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 10) {
        int mesh_dim = 0;
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_con_dims[0]}, {mesh_dim}, device_mesh_1d);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_con_dims[0]}, {mesh_dim}, device_mesh_1d);
        lhs_dev_dim[lhs_con_dims[0]] = mesh_dim;
        rhs_dev_dim[rhs_con_dims[0]] = mesh_dim;
        return std::make_pair(lhs_spec, rhs_spec);
      } else if (i == 11) {
        int batch_dim_idx = extract_var[0];
        int mesh_dim = 0;
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dims[batch_dim_idx]}, {mesh_dim},
                 device_mesh_1d);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_batch_dims[batch_dim_idx]}, {mesh_dim},
                 device_mesh_1d);
        lhs_dev_dim[lhs_batch_dims[batch_dim_idx]] = mesh_dim;
        rhs_dev_dim[rhs_batch_dims[batch_dim_idx]] = mesh_dim;
        out_dev_dim[lhs_batch_dims[batch_dim_idx]] = mesh_dim;
        return std::make_pair(lhs_spec, rhs_spec);
      } else {
        LOG(FATAL) << "can not parse the sharding string: " << stra_name;
      }
      matched = true;
      break;
    }
  }
  if (!matched) {
    LOG(FATAL) << "can not parse the sharding string: " << stra_name;
    HloSharding lhs_spec =
        Tile(lhs->shape(), {lhs_batch_dims[0]}, {0}, device_mesh);
    HloSharding rhs_spec =
        Tile(rhs->shape(), {rhs_batch_dims[0]}, {0}, device_mesh);
    return std::make_pair(lhs_spec, rhs_spec);
  }
}

static std::vector<int> infer_dev_dim_from_dep(const std::vector<int>& dev_dim,
                                               const std::vector<int>& deps) {
  std::vector<int> ret(dev_dim.size(), -1);
  for (int i = 0; i < deps.size(); i++) {
    ret[i] = dev_dim[deps[i]];
  }
  return ret;
}

static std::vector<int> infer_lrhs_dev_dim_from_dep(
    const HloInstruction* dot_f, const HloInstruction* fs,
    std::vector<int>& forward_lhs, std::vector<int>& forward_rhs,
    std::vector<int>& bs_to_fs) {
  if (dot_f->operand(0) == fs) {
    std::vector<int> input_bs(forward_lhs.size(), -1);
    for (int i = 0; i < bs_to_fs.size(); i++) {
      input_bs[i] = forward_lhs[bs_to_fs[i]];
    }
    return input_bs;
  } else {
    std::vector<int> input_bs(forward_rhs.size(), -1);
    for (int i = 0; i < bs_to_fs.size(); i++) {
      input_bs[i] = forward_rhs[bs_to_fs[i]];
    }
    return input_bs;
  }
}

// dst sharding to src sharding
static std::vector<int> infer_tile_assignment_from_dep(HloSharding& sharding,
                                                       std::vector<int> deps) {
  // get tile_assignment for sharding
  auto tile_assignment = sharding.tile_assignment();
  std::vector<int> source_tile_assignment(tile_assignment.dimensions().size());
  if (sharding.IsReplicated()) {
    source_tile_assignment[0] = 0;
    return source_tile_assignment;
  }
  int replicated_dim_num = 0;
  if (sharding.HasPartialReplication()) {
    // LOG(INFO) << "[NOTE]: sharding has partial replication";
    replicated_dim_num = 1;
    // append a replicated dim to the end
    source_tile_assignment[tile_assignment.dimensions().size() - 1] =
        tile_assignment.dimensions()[tile_assignment.dimensions().size() - 1];
  }
  for (int i = 0; i < source_tile_assignment.size() - replicated_dim_num; ++i) {
    source_tile_assignment[i] = tile_assignment.dimensions()[deps[i]];
  }
  return source_tile_assignment;
}

static std::vector<int> infer_arg_tile_assignment_from_dep(
    HloSharding& sharding, std::vector<int> deps) {
  // get tile_assignment for sharding
  auto tile_assignment = sharding.tile_assignment();
  if (tile_assignment.dimensions().size() == 1 &&
      tile_assignment.dimensions()[0] == 0) {
    return std::vector<int>(1, 0);
  }
  std::vector<int> source_tile_assignment(deps.size());

  for (int i = 0; i < source_tile_assignment.size(); ++i) {
    if (deps[i] == -1 || deps[i] == -2) {
      source_tile_assignment[i] = deps[i];
      continue;
    }
    source_tile_assignment[i] = tile_assignment.dimensions()[deps[i]];
  }
  return source_tile_assignment;
}

int infer_strategy_from_input_sharding(const HloInstruction* dot_b,
                                       const HloInstruction* bs,
                                       const StrategyMap& strategy_map,
                                       const ClusterEnvironment& cluster_env,
                                       const std::vector<int>& lhs_tile_assign,
                                       const std::vector<int>& rhs_tile_assign,
                                       const std::vector<int>& input_bs_dev_dim,
                                       bool is_dot) {
  // find dot strategy that match lhs&rhs tile assignment
  std::vector<ShardingStrategy> strategies =
      strategy_map.at(dot_b)->leaf_vector;
  auto is_vectors_equal = [](const std::vector<int>& v1,
                             const std::vector<int>& v2) {
    if (v1.size() != v2.size()) {
      return false;
    }
    return std::equal(v1.begin(), v1.end(), v2.begin());
  };
  PRINT_VECTOR_IF_DEBUG("DEBUG_FB_STRA_MATCHING",
                        "input lhs sharding: ", lhs_tile_assign,
                        "input rhs sharding: ", rhs_tile_assign);
  for (int stra_idx = 0; stra_idx < strategies.size(); ++stra_idx) {
    auto stra_name = strategies[stra_idx].name;
    HloSharding first = HloSharding::Replicate();
    HloSharding second = HloSharding::Replicate();
    std::pair<HloSharding, HloSharding> operand_sharding =
        std::make_pair(first, second);
    std::vector<int> lhs_dev_dim(lhs_tile_assign.size(), -1);
    std::vector<int> rhs_dev_dim(rhs_tile_assign.size(), -1);
    std::vector<int> out_placeholder;
    if (is_dot) {
      operand_sharding = get_operand_sharding_from_dot(
          dot_b, stra_name, cluster_env, lhs_dev_dim, rhs_dev_dim,
          out_placeholder);
    } else {
      operand_sharding = get_operand_sharding_from_conv(
          dot_b, stra_name, cluster_env, lhs_dev_dim, rhs_dev_dim,
          out_placeholder);
    }
    std::vector<int> current_lhs_tile_assign;
    std::vector<int> current_rhs_tile_assign;
    for (auto i : operand_sharding.first.tile_assignment().dimensions()) {
      current_lhs_tile_assign.push_back(int(i));
    }
    for (auto i : operand_sharding.second.tile_assignment().dimensions()) {
      current_rhs_tile_assign.push_back(int(i));
    }
    PRINT_VECTOR_IF_DEBUG(
        "DEBUG_FB_STRA_MATCHING",
        "matching: current lhs sharding: ", current_lhs_tile_assign,
        "matching: current rhs sharding: ", current_rhs_tile_assign);

    std::vector<int> need_to_match =
        dot_b->operand(0) == bs ? lhs_dev_dim : rhs_dev_dim;
    if (is_vectors_equal(current_lhs_tile_assign, lhs_tile_assign) &&
        is_vectors_equal(current_rhs_tile_assign, rhs_tile_assign)) {
      std::vector<int> bs_sharding =
          dot_b->operand(0) == bs ? lhs_tile_assign : rhs_tile_assign;
      auto check_bs_sharding = [&](std::vector<int> bs_sharding,
                                   std::vector<int> input_bs_dev_dim,
                                   std::vector<int> need_to_match) -> bool {
        for (int i = 0; i < bs_sharding.size(); i++) {
          if (bs_sharding[i] > 1 && input_bs_dev_dim[i] != need_to_match[i]) {
            return false;
          }
        }
        return true;
      };
      if (check_bs_sharding(bs_sharding, input_bs_dev_dim, need_to_match)) {
        return stra_idx;
      } else {
        PRINT_VECTOR_IF_DEBUG("DEBUG_DEV_DIM", "[NOTE]: Input bs_dev_dim",
                              input_bs_dev_dim, "not matched dev_dim",
                              need_to_match);
      }
    }
  }
  return -1;
}

static std::vector<int> build_bs_to_fs(
    std::vector<HloInstruction*>& to_share_op,
    std::vector<HloInstruction*>& from_share_op) {
  int share_op_dim = to_share_op.front()->shape().dimensions_size();
  std::vector<int> fs_share_dep(share_op_dim);
  std::vector<int> bs_share_dep(share_op_dim);
  dim_dep_from_path_for_bw(to_share_op, fs_share_dep);
  dim_dep_from_path_for_bw(from_share_op, bs_share_dep);
  PRINT_VECTOR_IF_DEBUG("DEBUG_DEPS", "fs_share_dep: ", fs_share_dep,
                        "bs_share_dep: ", bs_share_dep);
  return infer_dim_dep_from_shared_op(fs_share_dep, bs_share_dep);
}

static std::vector<std::string> get_dim_labels_from_conv(
    const HloInstruction* conv) {
  std::string conv_ins_str = conv->ToString();
  std::string start_str = "dim_labels=";
  std::string end_str = ",";
  size_t start_pos = conv_ins_str.find(start_str);
  size_t end_pos = conv_ins_str.find(end_str, start_pos);
  std::string dim_labels = conv_ins_str.substr(
      start_pos + start_str.length(), end_pos - start_pos - start_str.length());

  std::vector<std::string> parts;
  size_t pos = 0, last_pos = 0;
  while ((pos = dim_labels.find("_", last_pos)) != std::string::npos) {
    parts.push_back(dim_labels.substr(last_pos, pos - last_pos));
    last_pos = pos + 1;
  }
  parts.push_back(
      dim_labels.substr(last_pos, dim_labels.find("->") - last_pos));
  parts.push_back(dim_labels.substr(dim_labels.find("->") + 2));
  return parts;
}

static std::vector<int> resolve_deps_for_same_dst_with_complement(
    const std::vector<int>& bn_to_fs, const std::vector<int>& fo_to_fs) {
  // bn->fs, fo->fs, infer bn->fo

  int bs_dim = bn_to_fs.size();
  std::vector<int> bn_to_fo(bs_dim, -1);
  // cases like [0 1 -1], [0 -1 2] in MoE.
  // auto checkFunc = [](const std::vector<int>& vec) {
  //   for (int i = 0; i < vec.size(); i++) {
  //     if (vec[i] != i && vec[i] != -1) {
  //       return false;
  //     }
  //   }
  //   return true;
  // };
  // if (checkFunc(bn_to_fs) && checkFunc(fo_to_fs)) {
  //   std::iota(bn_to_fo.begin(), bn_to_fo.end(), 0);
  //   return bn_to_fo;
  // }
  int skipped = 0;
  int skipped_idx = -1;
  for (int i = 0; i < bs_dim; ++i) {
    if (bn_to_fs[i] != -1) {
      bn_to_fo[i] = std::find(fo_to_fs.begin(), fo_to_fs.end(), bn_to_fs[i]) -
                    fo_to_fs.begin();
    } else {
      skipped++;
      skipped_idx = i;
    }
  }
  if (skipped > 1) {
    LOG(FATAL) << "skipped more than 1 dims";
  } else if (skipped == 1) {
    bn_to_fo[skipped_idx] =
        ((bs_dim - 1) * bs_dim) / 2 -
        (std::accumulate(bn_to_fo.begin(), bn_to_fo.end(), 0) + 1);
  }
  return bn_to_fo;
}

static std::vector<int> build_conv_bn_to_fo(
    HloInstruction* dot_forward, HloInstruction* dot_backward,
    const HloInstruction* fs, const HloInstruction* bs,
    std::vector<int>& bs_to_fs, std::vector<HloInstruction*>& to_share_op,
    std::vector<HloInstruction*>& from_share_op) {
  int bs_dim = bs->shape().dimensions_size();
  // -------------build fo_to_fs----------------
  std::vector<std::string> parts = get_dim_labels_from_conv(dot_forward);
  auto findPositions = [](const std::string& first_part,
                          const std::string& second_part) -> std::vector<int> {
    std::vector<int> positions(second_part.size(), -1);
    for (int i = 0; i < second_part.size(); ++i) {
      auto ch = second_part[i];
      int pos = first_part.find(ch);
      if (pos == std::string::npos) {
        positions[i] = -1;
      } else {
        positions[i] = (pos);
      }
    }
    return positions;
  };
  std::replace(parts[2].begin(), parts[2].end(), 'f', 'o');
  std::vector<int> fo_to_fs = findPositions(
      dot_forward->operands()[0] == fs ? parts[0] : parts[1], parts[2]);

  // ------build deps for bn -> bs ----------------
  std::vector<int> bn_to_bs;
  std::vector<std::string> b_parts = get_dim_labels_from_conv(dot_backward);
  std::replace(b_parts[1].begin(), b_parts[1].end(), 'i', 'f');
  if (dot_backward->operands()[0] == bs) {
    bn_to_bs = findPositions(b_parts[0], b_parts[1]);
  } else {
    bn_to_bs = findPositions(b_parts[1], b_parts[0]);
  }

  // -----build deps for bn -> fs
  for (auto& idx : bn_to_bs) {
    if (idx != -1) {
      idx = bs_to_fs[idx];
    }
  }

  // -----build deps for bn -> fo
  std::vector<int> bn_to_fo =
      resolve_deps_for_same_dst_with_complement(bn_to_bs, fo_to_fs);
  return bn_to_fo;
}

static std::vector<int> build_dot_bn_to_fo(
    HloInstruction* dot_forward, HloInstruction* dot_backward,
    const HloInstruction* fs, const HloInstruction* bs,
    std::vector<int>& bs_to_fs, std::vector<HloInstruction*>& to_share_op,
    std::vector<HloInstruction*>& from_share_op) {
  int bs_dim = bs->shape().dimensions_size();
  const DotDimensionNumbers& ddn_f = dot_forward->dot_dimension_numbers();
  const DotDimensionNumbers& ddn_b = dot_backward->dot_dimension_numbers();

  // -------------build fo_to_fs----------------
  std::vector<int> fo_to_fs;
  if (dot_forward->operands()[0] == fs) {
    fo_to_fs.assign(ddn_f.lhs_batch_dimensions().begin(),
                    ddn_f.lhs_batch_dimensions().end());
    int lhs_remain_dim_index =
        RemainDimIndexForDotOp(bs_dim, ddn_f.lhs_batch_dimensions(),
                               ddn_f.lhs_contracting_dimensions());
    fo_to_fs.push_back(lhs_remain_dim_index);
    fo_to_fs.push_back(-1);
  } else {
    fo_to_fs.assign(ddn_f.rhs_batch_dimensions().begin(),
                    ddn_f.rhs_batch_dimensions().end());
    int rhs_remain_dim_index =
        RemainDimIndexForDotOp(bs_dim, ddn_f.rhs_batch_dimensions(),
                               ddn_f.rhs_contracting_dimensions());
    fo_to_fs.push_back(-1);
    fo_to_fs.push_back(rhs_remain_dim_index);
  }

  // ------build deps for bn -> bs -> fs ----------------
  std::vector<int> bn_to_bs(bs_dim, -1);
  auto asign_dims = [&](const auto& first_batch, const auto& first_contract,
                        const auto& second_batch, const auto& second_contract) {
    for (int i = 0; i < first_batch.size(); ++i) {
      bn_to_bs[first_batch[i]] = second_batch[i];
    }
    for (int i = 0; i < first_contract.size(); ++i) {
      bn_to_bs[first_contract[i]] = second_contract[i];
    }
  };
  if (dot_backward->operands()[0] == bs) {  // bn is rhs of dot_b
    // asign batch_dims for bn and bs
    asign_dims(ddn_b.rhs_batch_dimensions(), ddn_b.rhs_contracting_dimensions(),
               ddn_b.lhs_batch_dimensions(),
               ddn_b.lhs_contracting_dimensions());
  } else {  // bn is lhs of dot_b
    asign_dims(ddn_b.lhs_batch_dimensions(), ddn_b.lhs_contracting_dimensions(),
               ddn_b.rhs_batch_dimensions(),
               ddn_b.rhs_contracting_dimensions());
  }

  PRINT_VECTOR_IF_DEBUG("DEBUG_DEPS", "bn_to_bs: ", bn_to_bs);
  for (auto& idx : bn_to_bs) {
    if (idx != -1) {
      idx = bs_to_fs[idx];
    }
  }
  PRINT_VECTOR_IF_DEBUG("DEBUG_DEPS", "bn_to_fs: ", bn_to_bs);

  // -------build deps for bn -> fo ----------------
  std::vector<int> bn_to_fo =
      resolve_deps_for_same_dst_with_complement(bn_to_bs, fo_to_fs);
  return bn_to_fo;
}

static void BuildParallelPlanForBWDot(
    const HloInstructionSequence& sequence, const StrategyMap& strategy_map,
    const ClusterEnvironment& cluster_env, HloInstruction* dot_forward,
    HloInstruction* dot_backward, std::vector<HloInstruction*>& to_share_op,
    std::vector<HloInstruction*>& from_share_op, std::vector<int>& reindex) {
  // dot_f -> fs -> shared_op -> bs -> dot_b
  // dot_f -> fo -> bn ->dot_b
  // build dimension-level dependency for fs -> bs
  bool is_dot = dot_forward->opcode() == HloOpcode::kDot;
  std::reverse(to_share_op.begin(), to_share_op.end());
  to_share_op.pop_back();
  from_share_op.pop_back();
  const HloInstruction* fs = to_share_op.back();
  const HloInstruction* bs = from_share_op.back();
  // std::cerr << "build bs_to_fs for dot: " << strategy_map.at(dot_forward)->instruction_id << "\n";
  std::vector<int> bs_to_fs = build_bs_to_fs(to_share_op, from_share_op);

  // infer dimension-level dependence for fo and bn
  std::vector<int> bn_to_fo;
  if (is_dot) {
    bn_to_fo = build_dot_bn_to_fo(dot_forward, dot_backward, fs, bs, bs_to_fs,
                                  to_share_op, from_share_op);
  } else {
    // LOG(FATAL) << "conv case not implemented";
    bn_to_fo = build_conv_bn_to_fo(dot_forward, dot_backward, fs, bs, bs_to_fs,
                                   to_share_op, from_share_op);
  }
  TF_LOG_VECTOR(INFO, bs_to_fs, "bs_to_fs");
  TF_LOG_VECTOR(INFO, bn_to_fo, "bn_to_fo");
  PRINT_VECTOR_IF_DEBUG("DEBUG_DEPS", "*bn_to_fo", bn_to_fo);

  // check deps are valid
  const HloInstruction* bn = bs == dot_backward->operands()[0]
                                 ? dot_backward->operands()[1]
                                 : dot_backward->operands()[0];
  for (int i = 0; i < bn->shape().dimensions_size(); ++i) {
    CHECK_EQ(bn->shape().dimensions(i),
             dot_forward->shape().dimensions(bn_to_fo[i]));
  }

  // build dot strategy mapping relations from bs_to_fs, bn_to_fo
  // for each parallel strategies in dot_forward, delive sharding for fs and
  // fo, then propagate the sharding to bs and bn respectively refering to the
  // dim dependencies. then pick up the corresponding strategies in
  // dot_backward for shardings of bn and bs.
  // std::vector<int> value;
  for (auto& f_strategy : strategy_map.at(dot_forward)->leaf_vector) {
    auto stra_name = f_strategy.name;
    HloSharding first = HloSharding::Replicate();
    HloSharding second = HloSharding::Replicate();
    std::pair<HloSharding, HloSharding> operand_sharding =
        std::make_pair(first, second);
    std::vector<int> forward_lhs_dev_dim(
        dot_forward->shape().dimensions_size() + 1, -1);
    std::vector<int> forward_rhs_dev_dim(
        dot_forward->shape().dimensions_size() + 1, -1);
    std::vector<int> out_placeholder;
    if (is_dot) {
      operand_sharding = get_operand_sharding_from_dot(
          dot_forward, stra_name, cluster_env, forward_lhs_dev_dim,
          forward_rhs_dev_dim, out_placeholder);
    } else {
      // LOG(FATAL) << "conv case not implemented";
      operand_sharding = get_operand_sharding_from_conv(
          dot_forward, stra_name, cluster_env, forward_lhs_dev_dim,
          forward_rhs_dev_dim, out_placeholder);
    }
    forward_lhs_dev_dim.resize(
        operand_sharding.first.tile_assignment().dimensions().size());
    forward_rhs_dev_dim.resize(
        operand_sharding.second.tile_assignment().dimensions().size());

    std::vector<int> input_bs_dev_dim = infer_lrhs_dev_dim_from_dep(
        dot_forward, fs, forward_lhs_dev_dim, forward_rhs_dev_dim, bs_to_fs);

    // modify sharding of fs and fo to sharding of bn and bs
    const std::vector<int> bs_sharding = infer_tile_assignment_from_dep(
        dot_forward->operands()[0] == fs ? operand_sharding.first
                                         : operand_sharding.second,
        bs_to_fs);

    const std::vector<int> bn_sharding =
        infer_tile_assignment_from_dep(f_strategy.output_sharding, bn_to_fo);

    PRINT_VECTOR_IF_DEBUG("DEBUG_DEV_DIM",
                          "forward_lhs_dev_dim: ", forward_lhs_dev_dim,
                          "forward_rhs_dev_dim: ", forward_rhs_dev_dim,
                          "input_bs_dev_dim: ", input_bs_dev_dim);
    PRINT_STRING_IF_DEBUG(
        "DEBUG_FB_STRA_MATCHING", "STRATEGY: ", stra_name,
        "\n***LHS SHARDING: ", operand_sharding.first.ToString(),
        "\n***RHS SHARDING: ", operand_sharding.second.ToString(),
        "\n***OUTPUT SHARDING: ", f_strategy.output_sharding.ToString());
    PRINT_VECTOR_IF_DEBUG("DEBUG_FB_STRA_MATCHING",
                          "BACKWARD INPUT BS_SHARDING: ", bs_sharding,
                          "BACKWARD INPUT BN_SHARDING: ", bn_sharding);

    // find index
    int id_;
    if (dot_backward->operands()[0] == bs) {  // bs is lhs
      id_ = infer_strategy_from_input_sharding(
          dot_backward, bs, strategy_map, cluster_env, bs_sharding, bn_sharding,
          input_bs_dev_dim, is_dot);
    } else {
      id_ = infer_strategy_from_input_sharding(
          dot_backward, bs, strategy_map, cluster_env, bn_sharding, bs_sharding,
          input_bs_dev_dim, is_dot);
    }
    reindex.push_back(id_);
  }

  // Dump final forward and backward dot/conv strategies matching result
  // if (0) {
  //   std::cerr << "[SOLUTION FOR DOT: "
  //             << strategy_map.at(dot_forward)->instruction_id
  //             << " AND BACKWARD DOT: "
  //             << strategy_map.at(dot_backward)->instruction_id << " ]\n";
  //   for (int i = 0; i < value.size(); ++i) {
  //     if (value[i] != -1) {
  //       std::cerr << "  " << i << " "
  //                 << strategy_map.at(dot_forward)->leaf_vector[i].name << "
  //                 -> "
  //                 << value[i] << " "
  //                 <<
  //                 strategy_map.at(dot_backward)->leaf_vector[value[i]].name
  //                 << "\n";
  //     } else {
  //       std::cerr << "  " << i << " "
  //                 << strategy_map.at(dot_forward)->leaf_vector[i].name << "
  //                 -> "
  //                 << value[i] << " "
  //                 << "\n";
  //     }
  //   }
  // }
}

// build parallel blocks for each dot/conv instruction in forward and
// cluster its backward dot.
static void ClusterForwardBackwardDot(
    const HloInstructionSequence& sequence,
    const LeafStrategies& leaf_strategies, const StrategyMap& strategy_map,
    const ClusterEnvironment& cluster_env, std::vector<ParallelBlock>& parallel_blocks, int64_t loss_pos = -1) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  // int64_t loss_pos = 0;
  const char* specificated_loss_pos_str = std::getenv("LOSS_POS");
  if (specificated_loss_pos_str != nullptr){
    loss_pos = std::atoi(specificated_loss_pos_str);
  }
  if (loss_pos == -1){
    // [LLAMA FIXME]: temp for llama2.
    int64_t dot_count = 0;
    for (size_t instruction_id = 0; instruction_id < instructions.size();
       ++instruction_id) {
      const auto inst = instructions[instruction_id];
      if (inst->opcode() == HloOpcode::kDot) {
        dot_count++;
      }
    }
    std::cerr << "DOT COUNT IS " << dot_count << "\n";
    int layer_num = int(dot_count/(3*9));
    int64_t current_dot = 0;
    for (size_t instruction_id = 0; instruction_id < instructions.size();
       ++instruction_id) {
      const auto inst = instructions[instruction_id];
      if (inst->opcode() == HloOpcode::kDot) {
        current_dot++;
      }
      if ((current_dot) == layer_num * 9) {
        loss_pos = instruction_id - 3;
      }
    }
    std::cerr << "TEMP LOSS POS IS " << loss_pos << "\n";
  }
  
  absl::flat_hash_set<const HloInstruction*> visited;
  for (size_t instruction_id = 0; instruction_id < loss_pos; ++instruction_id) {
    const auto inst = instructions[instruction_id];
    if ((inst->opcode() == HloOpcode::kDot ||
         inst->opcode() == HloOpcode::kConvolution) &&
        visited.find(inst) == visited.end()) {
      // create new parallel block for each forward dot/conv.
      ParallelBlock p;
      int plan_size = strategy_map.at(inst)->leaf_vector.size();
      p.first.push_back(inst);
      std::vector<int> init_plan(plan_size);
      std::iota(init_plan.begin(), init_plan.end(), 0);
      p.second.push_back(init_plan);

      // std::cerr << "[INST]: "
      //           << inst->ToString(HloPrintOptions::ShortParsable()) << "\n";
      visited.insert(inst);
      // find backward dot for lhs and rhs operand, merge it to current pb.
      for (auto operand : inst->operands()) {
        std::vector<HloInstruction*> from_shared_operands;
        std::vector<HloInstruction*> to_shared_operands;
        HloInstruction* dot_backward = find_backward_dot(
            sequence, strategy_map, loss_pos, operand, visited, inst->shape(),
            to_shared_operands, from_shared_operands);
        if (dot_backward != nullptr) {
          visited.insert(dot_backward);
          to_shared_operands.insert(to_shared_operands.begin(), inst);
          const char* str_value = std::getenv("DEBUG_INST_ID");
          if (str_value != nullptr) {
            int value = std::stoi(str_value);
            if (value == instruction_id) {
              int dim_size = operand->shape().dimensions_size();
              std::cerr << "[dot_backward]: "
                        << dot_backward->ToString(
                               HloPrintOptions::ShortParsable())
                        << "\n";
              for (auto op : to_shared_operands) {
                std::cerr << "->" << HloOpcodeString(op->opcode());
                if (op->shape().dimensions_size() != dim_size) {
                  std::cerr << "*";
                }
              }
              std::cerr << "\n";
              for (auto op : from_shared_operands) {
                std::cerr << "->" << HloOpcodeString(op->opcode());
                if (op->shape().dimensions_size() != dim_size) {
                  std::cerr << "*";
                }
              }
              std::cerr << "\n";
            }
          }
          // build parallel plan for dot_backward.
          std::vector<int> strategies_reindex;
          BuildParallelPlanForBWDot(sequence, strategy_map, cluster_env, inst,
                                    dot_backward, to_shared_operands,
                                    from_shared_operands, strategies_reindex);
          CHECK_EQ(strategies_reindex.size(), plan_size);
          p.first.push_back(dot_backward);
          p.second.push_back(strategies_reindex);
        }
        dot_backward = nullptr;
      }
      // TODO: CHECK backward dots share the same grad input.
      parallel_blocks.push_back(p);
    }
  }
  // backward dot/conv that not merged into parallel_block, only in moe case for
  // now
  for (size_t instruction_id = loss_pos; instruction_id < instructions.size();
       ++instruction_id) {
    const auto inst = instructions[instruction_id];
    if ((inst->opcode() == HloOpcode::kDot ||
         inst->opcode() == HloOpcode::kConvolution) &&
        visited.find(inst) == visited.end()) {
      ParallelBlock p;
      int plan_size = strategy_map.at(inst)->leaf_vector.size();
      p.first.push_back(inst);
      std::vector<int> init_plan(plan_size);
      std::iota(init_plan.begin(), init_plan.end(), 0);
      p.second.push_back(init_plan);
      visited.insert(inst);
      parallel_blocks.push_back(p);
    }
  }
}

static void MergeParallelBlocks(std::vector<ParallelBlock>& parallel_blocks,
                                int src_pb_id, int dst_pb_id,
                                std::vector<int> reindexing) {
  // add inst in src pb to dst pb
  ParallelBlock& src_pb = parallel_blocks[src_pb_id];
  ParallelBlock& dst_pb = parallel_blocks[dst_pb_id];
  for (int i = 0; i < src_pb.first.size(); ++i) {
    dst_pb.first.push_back(src_pb.first[i]);
    std::vector<int> new_plan;
    for (auto idx : reindexing) {
      new_plan.push_back(src_pb.second[i][idx]);
    }
    dst_pb.second.push_back(new_plan);
  }
  src_pb.first.clear();
  src_pb.second.clear();
}

static void MergeInstToParallelBlocks(
    const HloInstructionSequence& sequence,
    std::vector<ParallelBlock>& parallel_blocks, int pb_idx, int inst_id,
    std::vector<int> reindexing) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  HloInstruction* inst = instructions[inst_id];
  parallel_blocks[pb_idx].first.push_back(inst);
  parallel_blocks[pb_idx].second.push_back(reindexing);
}



static void parseDotPairs(const std::string& input, DotPairs& dot_pairs, int offset_a = 0, int offset_b = 0) {
    std::istringstream ss(input);
    std::string segment;
    
    while (std::getline(ss, segment, ',')) {  
        std::istringstream segmentStream(segment);
        int a, b;
        segmentStream >> a >> b; 

        std::vector<int> vec;
        int value;
        while (segmentStream >> value) {  
            vec.push_back(value);
        }
        dot_pairs.push_back(std::make_pair(std::make_pair(a+offset_a, b+offset_b), vec));
    }

    return;
}

const std::string TRANSFORMER_DOT_PAIRS = "1 0 4 3 4, 2 0 4 3 4";
const std::string MOE_DOT_PAIRS = "1 0 4 3 4, 2 0 4 3 4, 5 4 3 3 3, 8 4 3 3 3, 7 6 2 0 1 3";
const std::string LLAMA_DOT_PAIRS = "1 0 0 1 2, 3 0 0 1 2, 2 0 3 4 3, 4 0 3 4 3, 7 6 0 1 2";

static void get_segment_dot_pair(std::vector<ParallelBlock>& parallel_blocks, DotPairs& dot_pairs) {
    if (parallel_blocks.size() == 6) { 
        parseDotPairs(TRANSFORMER_DOT_PAIRS, dot_pairs);
    } 
    else if (parallel_blocks.size() == 9) { 
        bool has_bmm = false;
        for (const auto& pb : parallel_blocks) {
            if (!pb.second.empty() && pb.second[0].size() == 4) {
                has_bmm = true;
                break; 
            }
        }
        if (has_bmm) {
            parseDotPairs(MOE_DOT_PAIRS, dot_pairs);
        } else {
            parseDotPairs(LLAMA_DOT_PAIRS, dot_pairs);
        }
    }
}

static void get_global_dot_pair(std::vector<ParallelBlock>& parallel_blocks, AutoShardingSolverOption& solver_option, DotPairs& dot_pairs) {

  std::string dot_pair_str = "";
  int dot_per_layers = int(parallel_blocks.size()/solver_option.num_layers);
  if (dot_per_layers == 6){
    dot_pair_str = TRANSFORMER_DOT_PAIRS;
  }
  else if (dot_per_layers == 9){
    dot_pair_str = LLAMA_DOT_PAIRS;
  }
  else if (dot_per_layers == 15){
    dot_pair_str = MOE_DOT_PAIRS + ", " + "10 9 4 3 4, 11 9 4 3 4";
  }
  else {
    dot_pair_str = "";
  }
  for(int i=0; i<solver_option.num_layers; ++i){
    parseDotPairs(dot_pair_str, dot_pairs, i*dot_per_layers, i*dot_per_layers);
  }
  return;

}

static void generate_fused_dot_pair(std::vector<ParallelBlock>& parallel_blocks,
                               AutoShardingSolverOption& solver_option,
                               DotPairs& dot_pairs
                               ) {
  if(solver_option.mode == "analysis_segment" || solver_option.mode == "profile_segment") {
    get_segment_dot_pair(parallel_blocks, dot_pairs);
  }
  else {
    get_global_dot_pair(parallel_blocks, solver_option, dot_pairs);
  }
  return;
}

static void ClusterAdjacentDot(const HloInstructionSequence& sequence,
                               const LeafStrategies& leaf_strategies,
                               const StrategyMap& strategy_map,
                               const ClusterEnvironment& cluster_env,
                               const DotPairs& dot_pairs,
                               std::vector<ParallelBlock>& parallel_blocks,
                               AutoShardingSolverOption& solver_option) {

  DotPairs new_dot_pairs;
  if (dot_pairs.size() == 0){
    generate_fused_dot_pair(parallel_blocks, solver_option, new_dot_pairs);
  }else{
    new_dot_pairs = dot_pairs;
  }

  for (auto pairs : new_dot_pairs) {
    int src = pairs.first.first;
    int dst = pairs.first.second;
    MergeParallelBlocks(parallel_blocks, src, dst, pairs.second);
  }

  // std::cerr << "mode is " << solver_option.mode << "\n";
  // std::cerr << "parallel_block size is " << parallel_blocks.size() << " dot_pairs is ";
  // for (auto pairs : new_dot_pairs) {
  //   int src = pairs.first.first;
  //   int dst = pairs.first.second;
  //   std::cerr << src << "->" << dst << ", "; 
  // }
  // std::cerr << "\n";


  return;
}

// return parallel_block id if node is in one parallel_block, or final following
// a node in parallel_block.
static int InParallelBlock(const HloInstruction* node,
                           const HloInstructionSequence& sequence,
                           const StrategyMap& strategy_map,
                           const LeafStrategies& leaf_strategies,
                           const std::vector<ParallelBlock>& parallel_blocks) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  int64_t node_id = strategy_map.at(node)->instruction_id;
  auto p = GetParallelBlock(node, parallel_blocks);
  if (p.first != -1) {
    return p.first;
  }
  // check followings of node
  const HloInstruction* follow_dst = node;
  while (strategy_map.at(follow_dst)->following != nullptr) {
    // check follow_dst in a parallel block.
    follow_dst =
        instructions[strategy_map.at(follow_dst)->following->instruction_id];
  }
  auto pb = GetParallelBlock(follow_dst, parallel_blocks);
  if (pb.first != -1) {
    return pb.first;
  }
  return -1;
}

static void FindOneHopPathsForMultiDst(const HloInstruction* inst, const std::vector<HloInstruction*> dsts,
                                      std::vector<std::vector<const HloInstruction*>>& paths){
  if (!inst || dsts.empty()){
    return;
  }
  std::unordered_set<const HloInstruction*> visited;
  std::queue<const HloInstruction*> q;
  std::unordered_map<const HloInstruction*, std::vector<const HloInstruction*>> parent_map;  // Stores paths.

  q.push(inst);
  visited.insert(inst);
  parent_map[inst] = {};  // Starting node has no parent.

  while (!q.empty()) {
    const HloInstruction* node = q.front();
    q.pop();

    // If the current node is a destination node, add the path to paths.
    if (std::find(dsts.begin(), dsts.end(), node) != dsts.end()) {
      std::vector<const HloInstruction*> path = parent_map[node];  // Get the current path.
      path.push_back(node);
      paths.push_back(path);
      continue;  // Stop exploring further along this path.
    }

    // Explore users of the current node.
    for (const auto user : node->users()) {
      if (visited.count(user)) continue;  // Avoid revisiting nodes.
      
      // Check if the current user is not part of dst_list before continuing.
      if (std::find(dsts.begin(), dsts.end(), user) == dsts.end()) {
        parent_map[user] = parent_map[node];  // Copy current path.
        parent_map[user].push_back(node);     // Add current node to the path.
        visited.insert(user);  // Mark the node as visited.
        q.push(user);  // Enqueue for further exploration.
      } else {
        // If user is a destination, add the current path up to this user node and stop exploring.
        std::vector<const HloInstruction*> path = parent_map[node];
        path.push_back(node);
        path.push_back(user);
        paths.push_back(path);
        visited.insert(user);  // Mark as visited so we don't reprocess this path.
      }
    }
  }
}


// return path from inst to dst.
static void FindPath(HloInstruction* inst, const HloInstruction* dst,
                     std::vector<HloInstruction*>& path) {
  if (!inst || !dst) {
    return;
  }
  std::unordered_set<HloInstruction*> visited;
  std::queue<HloInstruction*> q;
  q.push(inst);
  std::unordered_map<HloInstruction*, HloInstruction*> parent;
  while (!q.empty()) {
    HloInstruction* node = q.front();
    q.pop();
    if (node == dst) {
      while (node != inst) {
        path.push_back(node);
        node = parent[node];
      }
      std::reverse(path.begin(), path.end());
    }
    for (auto user : node->users()) {
      if (!visited.count(user)) {
        visited.insert(user);
        parent[user] = node;
        q.push(user);
      }
    }
  }

  return;
}

// return the dimensional dependency from frist inst to last inst.
// broadcast, reduce,  pad.,
// reshape, reverse, transpose
// substract, multiply, add, divide, select, compare, exponet, rsqrt, maximum,
static void InferDepsFromPath(std::vector<HloInstruction*> path,
                              std::vector<int>& deps) {
  std::iota(deps.begin(), deps.end(), 0);
  if (path.size() < 3) {
    return;
  }
  path.pop_back();
  path.erase(path.begin());
  PRINT_VECTOR_IF_DEBUG("INFER_DEPS_FROM_PATH", "start deps: ", deps);
  for (auto inst : path) {
    int64_t current_dim_size = inst->shape().rank();
    std::vector<int> map_vector(current_dim_size, -1);
    if (inst->opcode() == HloOpcode::kTranspose) {
      for (int64_t i = 0; i < inst->dimensions().size(); ++i) {
        map_vector[i] = inst->dimensions()[i];
      }
    } else if (inst->opcode() == HloOpcode::kReverse) {
      std::iota(map_vector.begin(), map_vector.end(), 0);
      if (inst->dimensions().size() > 1){
        std::swap(map_vector[inst->dimensions()[0]],
                map_vector[inst->dimensions()[1]]);
      }
    } else if (inst->opcode() == HloOpcode::kReshape) {
      deps_for_reshape(inst->operands()[0]->shape(), inst->shape(), map_vector);
    } else if (inst->opcode() == HloOpcode::kBroadcast) {
      int count = 0;
      for (auto pos : inst->dimensions()) {
        map_vector[pos] = count;
        count++;
      }
    } else if (inst->opcode() == HloOpcode::kReduce) {
      std::vector<int> iota_vec(deps.size());
      std::iota(iota_vec.begin(), iota_vec.end(), 0);

      for (int i = inst->dimensions().size() - 1; i >= 0; i--) {
        // std::cerr << "i : " << i << "dimension_i: " <<inst->dimensions()[i]
        // <<std::endl;
        iota_vec.erase(iota_vec.begin() + inst->dimensions()[i]);
      }
      map_vector = iota_vec;
    } else {
      bool ok = check_op_kind(inst);
      auto stringify_path = [](const std::vector<HloInstruction*>& path) -> std::string {
        std::stringstream ss;
        for (const auto inst : path) {
          ss << inst->ToString(HloPrintOptions::ShortParsable()) << "\n";
        }
        return ss.str();
      };
      CHECK(ok) << "not supported op: " << inst->ToString(HloPrintOptions::ShortParsable()) 
      << " for path: " << stringify_path(path);
      std::iota(map_vector.begin(), map_vector.end(), 0);
    }
    
    std::vector<int> new_dep(inst->shape().rank(), -1);
    PRINT_VECTOR_IF_DEBUG("INFER_DEPS_FROM_PATH", "    map_vector", map_vector);
    for (int64_t i = 0; i < current_dim_size; ++i) {
      if (map_vector[i] != -1) {
        new_dep[i] = (deps[map_vector[i]]);
      } else {
        new_dep[i] = -2;
      }
    }
  // [128, 32, 8] -> [128, 128] -> [128, 32, 8]
  // [0, 1, 2]  -> [0, 1] -> [0, 1, 1] ---> [0, 1, 2]
  auto findDuplicate = [&new_dep](int index) {
        for (int i = 0; i < index; ++i) {
            if (new_dep[i] == new_dep[index]) {
                return i;
            }
        }
        return -1;
  };
  auto findClosest = [&new_dep](int v) {
    int N = new_dep.size();
        std::vector<int> missingNumbers;
        for (int i = 0; i < N; ++i) {
            if (std::find(new_dep.begin(), new_dep.end(), i) == new_dep.end()) {
                missingNumbers.push_back(i);
            }
        }

        int closest = -1;
        int minDifference = N;
        for (int number : missingNumbers) {
            int difference = std::abs(number - v);
            if (difference < minDifference) {
                minDifference = difference;
                closest = number;
            }
        }
        return closest;
  };
  if (inst->opcode() == HloOpcode::kReshape 
    && inst->shape().rank() - inst->operands()[0]->shape().rank() > 0){
    //std::cerr << "[WARRNING]: Reset deps for complex reshape...";
  for (int i = 0; i<new_dep.size(); ++i){
    int prev_idx = findDuplicate(i);
    if (prev_idx>0){
      // reset map_vector[i] 
      new_dep[i] = findClosest(new_dep[i]);
      break;
    }
  }
    PRINT_VECTOR_IF_DEBUG("INFER_DEPS_FROM_PATH", "Before reset", deps);
  }
    deps = new_dep;
    PRINT_VECTOR_IF_DEBUG("INFER_DEPS_FROM_PATH", "mapped_deps", deps);
  }
}

static void InferDepsToPBDot(const std::vector<int>& deps_inst_to_user,
                             const std::vector<int>& deps_dot_to_user,
                             std::vector<int>& deps_to_pb_dot) {
  // return deps from inst to dot in parallel block.
  for (int i = 0; i < deps_inst_to_user.size(); ++i) {
    int idx = deps_inst_to_user[i];
    if (idx < 0) {
      continue;
    }
    deps_to_pb_dot[idx] = deps_dot_to_user[i];
  }
}

static void replaceContinuous(std::vector<int>& nums) {
  int n = nums.size();
  int start = 0, end = 0, cur = nums[0];
  for (int i = 1; i < n; i++) {
    if (nums[i] == cur) {
      end = i;
    } else {
      if (end > start) {
        for (int j = start + 1; j <= end; j++) {
          nums[j] = -1;
        }
      }
      cur = nums[i];
      start = i;
      end = i;
    }
  }
  if (end > start) {
    for (int j = start + 1; j <= end; j++) {
      nums[j] = -1;
    }
  }
}

static bool check_lhs_rhs(HloInstruction* inst, HloInstruction* dot) {
  std::vector<HloInstruction*> path;
  FindPath(inst, dot, path);
  if (path.size() == 0 || path.back() != dot) {
    LOG(FATAL) << "No path between inst and dep dot.";
  }
  path.insert(path.begin(), inst);
  HloInstruction* operand = path[path.size() - 2];
  if (operand == dot->operand(0)) {
    return true;
  } else if (operand == dot->operand(1)) {
    return false;
  } else {
    LOG(FATAL) << "No path between inst and dep dot.";
  }
}

static int BuildDepsToPB(const HloInstructionSequence& sequence,
                         const StrategyMap& strategy_map, HloInstruction* inst,
                         int user_id, int& lhs_or_rhs, std::vector<int>& deps_to_pb_dot,
                         ParallelBlock& dest_pb,
                         DepsToDotMap& deps_to_dot_map) {
  // generate deps from inst to user_inst and deps from user_inst to one dot in
  // parallel_block.
  int dot_id = -1;
  int spec_side = -1;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  HloInstruction* user_inst = instructions[user_id];

  // get path from user_inst to dot.
  HloInstruction* follow_dst = user_inst;
  std::vector<HloInstruction*> following_path;
  following_path.push_back(follow_dst);
  while (strategy_map.at(follow_dst)->following != nullptr) {
    // check follow_dst in a parallel block.
    follow_dst =
        instructions[strategy_map.at(follow_dst)->following->instruction_id];
    following_path.push_back(follow_dst);
    if (std::find(dest_pb.first.begin(), dest_pb.first.end(), follow_dst) !=
        dest_pb.first.end()) {  // in parallel block.
      // if (follow_dst->opcode() != HloOpcode::kDot &&
      //     follow_dst->opcode() != HloOpcode::kConvolution) {
      //   CHECK_EQ(follow_dst->operand_count(), 1);
      //   int follow_dst_id =
      //       strategy_map.at(follow_dst->operand(0))->instruction_id;
      //   follow_dst = instructions[follow_dst_id];
      //   following_path.push_back(follow_dst);
      //   continue;
      // }
      break;
    }
  }
  // get path from inst to user_id
  std::vector<HloInstruction*> inst_to_user_path;
  FindPath(inst, user_inst, inst_to_user_path);
  inst_to_user_path.insert(inst_to_user_path.begin(), inst);
  std::reverse(following_path.begin(), following_path.end());
  if (following_path.front()->opcode() == HloOpcode::kDot ||
      following_path.front()->opcode() == HloOpcode::kConvolution) {
    dot_id = strategy_map.at(following_path.front())->instruction_id;
    if (user_id > dot_id) {
      // output
      lhs_or_rhs = 2;
    } else if (user_id == dot_id) {
      // lhs or rhs, check inst_to_user_path
      HloInstruction* target_dot = inst_to_user_path.back();
      HloInstruction* operand = inst_to_user_path[inst_to_user_path.size() - 2];
      if (target_dot->opcode() != HloOpcode::kDot &&
          target_dot->opcode() != HloOpcode::kConvolution) {
        LOG(FATAL) << "[BuildDepsToPB]: inst_to_user_path.back() is not a dot "
                      "instruction.";
      }
      if (operand == user_inst->operand(0)) {
        lhs_or_rhs = 0;
      } else if (operand == user_inst->operand(1)) {
        lhs_or_rhs = 1;
      } 
    } 
  } else {  // replace dot_id with following_path.front's dep_dot 
    dot_id = deps_to_dot_map.at(following_path.front()).first;
    if (strategy_map.at(following_path.front())->instruction_id > dot_id) {
      lhs_or_rhs = 2;
    } else {
      if (check_lhs_rhs(following_path.front(), instructions[dot_id])) {
	      lhs_or_rhs = 0;
      } else {
	      lhs_or_rhs = 1;
      }
    }
  }
  PRINT_STRING_IF_DEBUG("BUILD_DEPS_TO_PB", "[HANDLE DEPS FOR INST] ",
                        inst->ToString(HloPrintOptions::ShortParsable()));
  PRINT_STRING_IF_DEBUG("BUILD_DEPS_TO_PB", "[DOT_TO_USER_PATH] ", " ");

  for (auto inst : following_path) {
    if((inst->opcode() == HloOpcode::kDot) 
      && (inst!=following_path[0]&&inst!=following_path[following_path.size()-1])){
      return -2;
    }
    PRINT_STRING_IF_DEBUG("BUILD_DEPS_TO_PB", " ",
                          inst->ToString(HloPrintOptions::ShortParsable()));
  }
  PRINT_STRING_IF_DEBUG("BUILD_DEPS_TO_PB", "[INST_TO_USER_PATH] ", " ");
  for (auto inst : inst_to_user_path) {
    if((inst->opcode() == HloOpcode::kDot) 
      && (inst!=inst_to_user_path[0]&&inst!=inst_to_user_path[inst_to_user_path.size()-1])){
      return -2;
    }
    PRINT_STRING_IF_DEBUG("BUILD_DEPS_TO_PB", " ",
                          inst->ToString(HloPrintOptions::ShortParsable()));
  }
  // shape().dimensions_size()
  std::vector<int> deps_dot_to_user(
      following_path.front()->shape().dimensions_size());
  std::vector<int> deps_inst_to_user(inst->shape().dimensions_size());
  // std::cerr << "InferDeps: " << strategy_map.at(inst) -> instruction_id << " user: " << user_id << "\n";
  InferDepsFromPath(following_path, deps_dot_to_user);
  if (following_path.front()->opcode() != HloOpcode::kDot &&
      following_path.front()->opcode() != HloOpcode::kConvolution) {
    // brige deps_dot_to_user with
    // deps_to_dot_map.at(following_path.front()).second
    for (int idx = 0; idx < deps_dot_to_user.size(); idx++) {
      if (deps_dot_to_user[idx] < 0) {
        continue;
      }
      //  std::cerr << strategy_map.at(following_path.front())->instruction_id
      //            << " " << strategy_map.at(inst)->instruction_id << " "
      //            << deps_dot_to_user[idx] << " "
      //            << deps_to_dot_map.at(following_path.front())
      //                   .second[deps_dot_to_user[idx]]
      //            << "\n";
      deps_dot_to_user[idx] = deps_to_dot_map.at(following_path.front())
                                  .second[deps_dot_to_user[idx]];
    }
  }
  InferDepsFromPath(inst_to_user_path, deps_inst_to_user);
  // infer deps from inst to dot.
  InferDepsToPBDot(deps_inst_to_user, deps_dot_to_user, deps_to_pb_dot);
  // replace continuous deps with -1.
  replaceContinuous(deps_to_pb_dot);
  deps_to_dot_map[inst] = std::make_pair(dot_id, deps_to_pb_dot);

  PRINT_VECTOR_IF_DEBUG("BUILD_DEPS_TO_PB",
                        "inst_to_user: ", deps_inst_to_user);
  PRINT_VECTOR_IF_DEBUG("BUILD_DEPS_TO_PB", "dot_to_user: ", deps_dot_to_user);
  PRINT_VECTOR_IF_DEBUG("BUILD_DEPS_TO_PB", "[deps to dot]: ", deps_to_pb_dot);


  return dot_id;
}


//"SbSi = SbSi x SbR @ \\{(\\d+),(\\d+)\\}");
static void get_dev_dim(const ShardingStrategy& stra,
                        std::vector<int>& dev_dim) {
  // partition special case: reshape operator may have S01 @ {0, 1}
  auto stra_name = stra.name;
  std::regex pattern_0("S(\\d) @ (\\d)");
  std::regex pattern_1("S(\\d) @ (\\d) 1d");
  std::regex pattern_2("S(\\d)(\\d) @ \\{(\\d),(\\d)\\}");
  std::regex pattern_3("S\\{(\\d),(\\d)\\} @ \\{(\\d),(\\d)\\}");
  std::vector<std::regex> patterns = {pattern_0, pattern_1, pattern_2,
                                      pattern_3};
  bool matched = false;
  for (int i = 0; i < patterns.size(); ++i) {
    std::smatch match;
    if (std::regex_match(stra_name, match, patterns[i])) {
      std::vector<int> extract_var;
      for (int j = 1; j < match.size(); ++j) {
        extract_var.push_back(std::stoi(match[j].str()));
      }
      if (i == 0 || i == 1) {
        dev_dim[extract_var[0]] = extract_var[1];
      } else if (i == 2 || i == 3) {
        dev_dim[extract_var[0]] = extract_var[2];
        dev_dim[extract_var[1]] = extract_var[3];
      }
      matched = true;
      break;
    }
  }
}

static bool check_device_order(const ShardingStrategy& stra) {
  Array<int64_t> tile_assignment_ = stra.output_sharding.tile_assignment();
  std::vector<int> dims;
  for (const auto& str : tile_assignment_) {
    dims.push_back(str);
  }
  for (int i = 1; i < dims.size(); ++i) {
    if (dims[i] <= dims[i - 1]) {
      return false;
    }
  }
  return true;
}

// given the expected tile_assign and dev_dim, return the id of matched
// strategy.
static int infer_args_strategy_from_input_sharding(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const ClusterEnvironment& cluster_env, std::vector<int>& tile_assign,
    std::vector<int>& expected_dev_dim) {
  // -2, -1 indicate broadcast dimensions and cannot infered from deps
  // dimensions, replace them with 1.
  for (auto& tile : tile_assign) {
    tile = (tile == -2 || tile == -1) ? 1 : tile;
  }
  // if all dimensions tile assign are 1, simplify it to {0}.
  if (std::all_of(tile_assign.begin(), tile_assign.end(),
                  [](auto i) { return i == 1; })) {
    tile_assign = std::vector<int>{0};
  }
  PRINT_VECTOR_IF_DEBUG("INFER_ARG_STRATEGY",
                        "[EXPECTED TILE ASSIGN]: ", tile_assign);
  PRINT_VECTOR_IF_DEBUG("INFER_ARG_STRATEGY",
                        "[EXPECTED DEV DIM]: ", expected_dev_dim);

  // check output tile assignment.
  absl::flat_hash_set<std::vector<int>> visited_tile_assign;
  for (int i = 0; i < strategy_map.at(inst)->leaf_vector.size(); ++i) {
    auto stra = strategy_map.at(inst)->leaf_vector[i];
    int min_dim_size =
        std::min(tile_assign.size(),
                 static_cast<size_t>(inst->shape().dimensions_size()));

    // PRINT_VECTOR_IF_DEBUG("INFER_ARG_STRATEGY", "[TO MATCH TILE ASSIGN]: ",
    //                       to_match_tile_assign);

    bool tile_assign_matched = true;

    for (int j = 0; j < min_dim_size; ++j) {
      if (stra.output_sharding.tile_assignment().dimensions()[j] !=
          tile_assign[j]) {
        tile_assign_matched = false;
      }
    }
    if (!tile_assign_matched) {
      continue;
    }
    // R case
    if (tile_assign.size() == 1 && tile_assign[0] == 0) {
      PRINT_STRING_IF_DEBUG("INFER_ARG_STRATEGY", " Matched R ");
      return i;
    }
    // Sharding case, check device dim mapping
    std::vector<int> dev_dim(inst->shape().rank(), -1);
    get_dev_dim(stra, dev_dim);

    // compare, select: [tile assign] assume strategies with same tile
    // assignment are device mesh (0, 1) and (1, 0)
    bool all_neg_one = std::all_of(dev_dim.begin(), dev_dim.end(),
                                   [](int i) { return i == -1; });
    // used by compare and select
    if (all_neg_one) {
      if (inst->opcode() == HloOpcode::kSelect ||
          inst->opcode() == HloOpcode::kCompare) {
        std::vector<int> assigned_dev_dim;
        if (check_device_order(stra)) {
          assigned_dev_dim = {0, 1};
        } else {
          assigned_dev_dim = {1, 0};
        }
        auto first_dim = std::find_if(tile_assign.begin(), tile_assign.end(),
                                      [](int i) { return i > 1; });
        if (first_dim != tile_assign.end()) {
          dev_dim[std::distance(tile_assign.begin(), first_dim)] =
              assigned_dev_dim[0];
        }

        auto count = std::count_if(tile_assign.begin(), tile_assign.end(),
                                   [](int i) { return i > 1; });
        if (count >= 2) {
          auto second_dim = std::find_if(first_dim + 1, tile_assign.end(),
                                         [](int i) { return i > 1; });
          if (second_dim != tile_assign.end()) {
            dev_dim[std::distance(tile_assign.begin(), second_dim)] =
                assigned_dev_dim[1];
          }
        }
      } else {
        LOG(FATAL) << "[ERROR] "
                   << inst->ToString(HloPrintOptions::ShortParsable());
      }
    }

    PRINT_VECTOR_IF_DEBUG("INFER_ARG_STRATEGY",
                          "[TO MATCH DEV DIM]: ", dev_dim);

    bool dev_dim_match = true;
    for (int j = 0; j < dev_dim.size(); ++j) {
      if (tile_assign[j] != 1 && dev_dim[j] != expected_dev_dim[j] &&
          (cluster_env.device_mesh.dim(0) > 1 &&
           cluster_env.device_mesh.dim(1) > 1)) {
        dev_dim_match = false;
      }
    }
    if (dev_dim_match) {
      PRINT_STRING_IF_DEBUG("INFER_ARG_STRATEGY",
                            " Matched strategy: " + stra.name + " ");
      return i;
    }
  }
  PRINT_STRING_IF_DEBUG(
      "INFER_ARG_STRATEGY",
      "[WARRNING] Cannot match any strategy, using R to replace ");

  for (int i = 0; i < strategy_map.at(inst)->leaf_vector.size(); ++i) {
    auto stra = strategy_map.at(inst)->leaf_vector[i];
    if (stra.name == "R") {
      return i;
    }
  }
}

static void GenerateStrategiesReindexWithDepsToPB(
    const HloInstructionSequence& sequence, HloInstruction* inst,
    int dep_dot_id, int& lhs_or_rhs, const std::vector<int>& deps_to_pb_dot,
    const StrategyMap& strategy_map,
    std::vector<ParallelBlock>& parallel_blocks,
    std::vector<int>& strategies_reindex,
    const ClusterEnvironment& cluster_env) {
  // generate strategies_reindex with deps_to_pb_dot.
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  HloInstruction* dep_dot = instructions[dep_dot_id];
  bool is_dot = dep_dot->opcode() == HloOpcode::kDot;
  std::vector<int> inst_stra_reindex;
  HloSharding input_sharding = HloSharding::Replicate();
  std::vector<int> input_dev_dim;

  // infer sharding of inst from dep_dot input or output.
  // [FIXME]: need generate as parallel_plan for dep dot.
  for (auto& dot_stra : strategy_map.at(dep_dot)->leaf_vector) {
    // find a sharding of inst that match current dep_dot sharding under
    // deps_to_pb_dot.
    auto stra_name = dot_stra.name;

    PRINT_STRING_IF_DEBUG("DEBUG_REINDEX_DEPS_TO_PB",
                          " Try to match strategy: " + stra_name + " ");

    HloSharding first = HloSharding::Replicate();
    HloSharding second = HloSharding::Replicate();
    std::pair<HloSharding, HloSharding> operand_sharding =
        std::make_pair(first, second);
    std::vector<int> lhs_dev_dim(dep_dot->shape().dimensions_size() + 1, -1);
    std::vector<int> rhs_dev_dim(dep_dot->shape().dimensions_size() + 1, -1);
    std::vector<int> out_dev_dim(dep_dot->shape().dimensions_size() + 1, -1);

    if (is_dot) {
      operand_sharding =
          get_operand_sharding_from_dot(dep_dot, stra_name, cluster_env,
                                        lhs_dev_dim, rhs_dev_dim, out_dev_dim);
    } else {
      operand_sharding =
          get_operand_sharding_from_conv(dep_dot, stra_name, cluster_env,
                                         lhs_dev_dim, rhs_dev_dim, out_dev_dim);
    }
    lhs_dev_dim.resize(
        operand_sharding.first.tile_assignment().dimensions().size());
    rhs_dev_dim.resize(
        operand_sharding.second.tile_assignment().dimensions().size());
    out_dev_dim.resize(
        dot_stra.output_sharding.tile_assignment().dimensions().size());

    if (lhs_or_rhs == -1 && strategy_map.at(inst)->instruction_id < dep_dot_id) {
      // inst is an operand of dep_dot.
      std::cerr << "generateStrategies, check lhs or rhs: " << strategy_map.at(inst)->instruction_id << " -> " << strategy_map.at(dep_dot)->instruction_id << "\n";
      if (check_lhs_rhs(inst, dep_dot)) {
        input_sharding = operand_sharding.first;
        input_dev_dim = lhs_dev_dim;
      } else {
        input_sharding = operand_sharding.second;
        input_dev_dim = rhs_dev_dim;
      }
    } else if(lhs_or_rhs == -1 && strategy_map.at(inst)->instruction_id > dep_dot_id) {
      input_sharding = dot_stra.output_sharding;
      // how to get dev_dim from dot_stra output?
      input_dev_dim = out_dev_dim;
    } else if (lhs_or_rhs == 0) {
      input_sharding = operand_sharding.first;
      input_dev_dim = lhs_dev_dim; 
    } else if (lhs_or_rhs == 1) {
      input_sharding = operand_sharding.second;
      input_dev_dim = rhs_dev_dim; 
    } else {
      input_sharding = dot_stra.output_sharding;
      // how to get dev_dim from dot_stra output?
      input_dev_dim = out_dev_dim;
    }

    // source_tile_assignment[i] = tile_assignment.dimensions()[deps[i]];
    std::vector<int> expected_tile_assignment =
        infer_arg_tile_assignment_from_dep(input_sharding, deps_to_pb_dot);
    PRINT_STRING_IF_DEBUG("DEBUG_REINDEX_DEPS_TO_PB","input_sharding: ", input_sharding.ToString());

    std::vector<int> expected_dev_dim(deps_to_pb_dot.size());
    for (int i = 0; i < deps_to_pb_dot.size(); i++) {
      if (deps_to_pb_dot[i] == -1 || deps_to_pb_dot[i] == -2) {
        expected_dev_dim[i] = -1;
        continue;
      }
      expected_dev_dim[i] = input_dev_dim[deps_to_pb_dot[i]];
    }

    PRINT_VECTOR_IF_DEBUG(
        "DEBUG_REINDEX_DEPS_TO_PB", "-------deps to pb dot: ", deps_to_pb_dot,
        "-------expected_tile_assignment: ", expected_tile_assignment,
        "-------expected_dev_dim: ", expected_dev_dim);
    int id = infer_args_strategy_from_input_sharding(
        inst, strategy_map, cluster_env, expected_tile_assignment,
        expected_dev_dim);
    int loss_pos;
    const char* specificated_loss_pos_str = std::getenv("LOSS_POS");
    if (specificated_loss_pos_str != nullptr){
      loss_pos = std::atoi(specificated_loss_pos_str);
    }else {
      loss_pos = instructions.size();
    } 
    //HUWF[0826]
    // if (inst->opcode() == HloOpcode::kReshape && dep_dot->shape().dimensions_size() > 2 && strategy_map.at(inst)->instruction_id < loss_pos){
    //   //std::cerr << strategy_map.at(inst)->instruction_id << ": Force set reshape related bmm to R.\n";
    //   for (int i = 0; i < strategy_map.at(inst)->leaf_vector.size(); ++i) { 
    //     auto tile_assign = strategy_map.at(inst)->leaf_vector[i].output_sharding.tile_assignment().dimensions();
    //     if (tile_assign.size() == 1 && tile_assign[0] == 0) {
    //       id = i;
    //     }
    //   }
    // }
    strategies_reindex.push_back(id);
  }
}


static void GenerateReindexForMultiPBUserInst(
  const HloInstructionSequence& sequence, HloInstruction* inst,
  const StrategyMap& strategy_map, const LeafStrategies& leaf_strategies, const ClusterEnvironment& cluster_env, 
  std::vector<ParallelBlock>& parallel_blocks, DepsToDotMap& deps_to_dot_map, 
  std::vector<std::pair<const HloInstruction*, int>>& parallel_block_users, ReindexItemList& reindex_arr){
  std::ostringstream os;
  
  ReindexItem multiple_reindex;
  absl::flat_hash_map<int, std::vector<const HloInstruction*>> multi_user_pb;
  multiple_reindex.set_inst_leaf_id(strategy_map.at(inst)->id);
  const StrategyVector* inst_stras = leaf_strategies[multiple_reindex.inst_leaf_id()]; 
  multiple_reindex.set_inst_id(inst_stras->instruction_id);
  multiple_reindex.set_stra_num(inst_stras->leaf_vector.size());
  multiple_reindex.set_is_key_inst(false);
  multiple_reindex.set_deps_by_multi_pb(true);

  for (auto p_ : parallel_block_users) {
    if (multi_user_pb.find(p_.second) == multi_user_pb.end()) {
      multi_user_pb[p_.second] = std::vector<const HloInstruction*>();
    }
    multi_user_pb[p_.second].push_back(p_.first);
  }

  // set deps dot and corresponding reindex for each dependent parallel block.
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  os << "Handel multiple pb user for inst: " << strategy_map.at(inst) -> instruction_id << " ";
  for (auto kvp : multi_user_pb) {
    auto user_insts = kvp.second;
    auto nearest_user = std::min_element(
        user_insts.begin(), user_insts.end(),
        [&](const HloInstruction* a, const HloInstruction* b) {
          return strategy_map.at(a)->instruction_id <
                              strategy_map.at(b)->instruction_id;
        });
    std::vector<int> deps_to_pb_dot_tmp(inst->shape().dimensions_size(), -1);
    int lhs_or_rhs = -1;
    std::vector<int> reindexing;
    int user_inst_id = strategy_map.at(*nearest_user)->instruction_id;
    
    // find deps dot's instruction id.
    int dep_dot_id = BuildDepsToPB(
        sequence, strategy_map, inst,
        user_inst_id, lhs_or_rhs,
        deps_to_pb_dot_tmp, parallel_blocks[kvp.first], deps_to_dot_map);
    if (dep_dot_id == -2) 
    {
      os << "[GenerateReindexForMultiPBUserInst]:Ignore inst " << strategy_map.at(inst)->instruction_id << " and its user " << user_inst_id << "\n";
      continue;
    }
    const HloInstruction* first_dot_in_target_pb = parallel_blocks[kvp.first].first[0];
    multiple_reindex.add_deps_pb_id(strategy_map.at(first_dot_in_target_pb)->id);
    // generate reindex
    ReindexData* reindex_data = multiple_reindex.add_reindexing();
    GenerateStrategiesReindexWithDepsToPB(
        sequence, inst, dep_dot_id, lhs_or_rhs, deps_to_pb_dot_tmp,
        strategy_map, parallel_blocks, reindexing, cluster_env);

    // need futhur reindexing if the dep_dot is not the first dot in it's parallel block.
    std::vector<int> new_strategies_reindex = reindexing;
    auto pb_idx_inst_idx = GetParallelBlock(instructions[dep_dot_id], parallel_blocks);
    os << " pb idx " << pb_idx_inst_idx.first << " reindex: ";
    std::vector<int>& plan = parallel_blocks[kvp.first].second[pb_idx_inst_idx.second];
    for (int i = 0; i < reindexing.size(); ++i) {
      if (plan[i] > -1 && plan[i] < reindexing.size()) {
        new_strategies_reindex[i] = reindexing[plan[i]];
      }
      os<< new_strategies_reindex[i] << " ";
    }
    for (auto r : new_strategies_reindex){
      reindex_data->add_reindex(r);
    }
  }
  reindex_arr.mutable_reindex_items()->Add(std::move(multiple_reindex));
  os << "\n";
  LOG(INFO) << os.str();

}

static std::pair<int, std::vector<int>> AnalysisDependenceToPB(
    const HloInstructionSequence& sequence, HloInstruction* inst,
    const LeafStrategies& leaf_strategies, const StrategyMap& strategy_map,
    const ClusterEnvironment& cluster_env, std::vector<ParallelBlock>& parallel_blocks, 
    const AutoShardingSolverOption& solver_option, DepsToDotMap& deps_to_dot_map, ReindexItemList& reindex_arr, ParallelBlockConfig pb_config) {
  // check all users of this inst that not followed this inst lied in the same
  // parallel block.
  int pb = -1;
  std::vector<int> strategies_reindex;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  if (!inst) {
    return std::make_pair(pb, strategies_reindex);
  }
  int inst_id = strategy_map.at(inst)->instruction_id;

  const char* str_value = std::getenv("DEBUG_INST_ID");
  if (str_value != nullptr) {
    int value = std::stoi(str_value);
    if (inst_id == value) {
      std::cerr << "--------------------DEBUG_INST_ID: " << inst_id
                << "-------------------------------\n";
      setenv("BUILD_DEPS_TO_PB", "true", 1);
      setenv("DEBUG_REINDEX_DEPS_TO_PB", "true", 1);
      setenv("INFER_ARG_STRATEGY", "true", 1);
      setenv("INFER_DEPS_FROM_PATH", "true", 1);
    }
  }
  std::unordered_set<const HloInstruction*> visited;
  std::queue<const HloInstruction*> q({inst});
  std::vector<std::pair<const HloInstruction*, int>> parallel_block_users;
  while (!q.empty()) {
    const HloInstruction* node = q.front();
    q.pop();
    int pb_id = InParallelBlock(node, sequence, strategy_map, leaf_strategies,
                                parallel_blocks);
    if (pb_id != -1) {
      parallel_block_users.push_back(std::make_pair(node, pb_id));
      continue;
    }
    // A dot not in any parallel_block, stop searching, reshape.53 in moe 1d.
    if (node->opcode() == HloOpcode::kDot ||
        node->opcode() == HloOpcode::kConvolution) {
      parallel_block_users.push_back(std::make_pair(node, -2));
      continue;
    }
    for (auto user : node->users()) {
      if (!visited.count(user)) {
        visited.insert(user);
        q.push(user);
      }
    }
  }
  if (parallel_block_users.size() == 0) {
    return std::make_pair(pb, strategies_reindex);
  }
  // check if all parallel block users are in the same parallel block.
  // return true if all second values in parallel_block_users are same.
  auto all_same =
      [](const std::vector<std::pair<const HloInstruction*, int>>& p) {
        return std::all_of(p.begin(), p.end(),
                           [&](const std::pair<const HloInstruction*, int>& i) {
                             return i.second == p.front().second;
                           });
      };
  if (!all_same(parallel_block_users)){
      GenerateReindexForMultiPBUserInst(sequence, inst, strategy_map, leaf_strategies,
          cluster_env, parallel_blocks, deps_to_dot_map, 
          parallel_block_users, reindex_arr);
    //}
    return std::make_pair(pb, strategies_reindex);
  }
  // analysis dependence to this parallel block and generate reindex strategy.
  int pb_idx = parallel_block_users.front().second;
  if (pb_idx < 0) {
    return std::make_pair(pb, strategies_reindex);
  }
  // reindex strategy, analysis inter-inst dependence between inst and its
  // nearest pb users. for each user in parallel_block_users, check the inst
  // indicated by user.first, check the ins's id by using
  // strategy_map.at(ins)->instruction_id, and return the samllest one.
  auto nearest_user =
      std::min_element(parallel_block_users.begin(), parallel_block_users.end(),
                       [&](const std::pair<const HloInstruction*, int>& a,
                           const std::pair<const HloInstruction*, int>& b) {
                         return strategy_map.at(a.first)->instruction_id <
                                strategy_map.at(b.first)->instruction_id;
                       });
  int nearest_user_id = strategy_map.at(nearest_user->first)->instruction_id;
  std::vector<int> deps_to_pb_dot(inst->shape().dimensions_size(), -1);

  int lhs_or_rhs = -1;
  int dep_dot_id =
      BuildDepsToPB(sequence, strategy_map, inst, nearest_user_id, lhs_or_rhs, 
                    deps_to_pb_dot, parallel_blocks[pb_idx], deps_to_dot_map);
  GenerateStrategiesReindexWithDepsToPB(
      sequence, inst, dep_dot_id, lhs_or_rhs, deps_to_pb_dot, strategy_map, parallel_blocks,
      strategies_reindex, cluster_env);

  std::vector<int> new_strategies_reindex = strategies_reindex;
  auto pb_idx_inst_idx =
      GetParallelBlock(instructions[dep_dot_id], parallel_blocks);
  std::vector<int>& plan =
      parallel_blocks[pb_idx].second[pb_idx_inst_idx.second];
  PRINT_VECTOR_IF_DEBUG("INFER_ARG_STRATEGY", "dot plan: ", plan);
  for (int i = 0; i < strategies_reindex.size(); ++i) {
    if (plan[i] > -1 && plan[i] < strategies_reindex.size()) {
      PRINT_VECTOR_IF_DEBUG("INFER_ARG_STRATEGY", "dot plan: ", plan);
      new_strategies_reindex[i] = strategies_reindex[plan[i]];
    }
  }
  PRINT_VECTOR_IF_DEBUG("INFER_ARG_STRATEGY", "stra: ", strategies_reindex);
  PRINT_VECTOR_IF_DEBUG("INFER_ARG_STRATEGY",
                        "new stra: ", new_strategies_reindex);

  if (str_value != nullptr) {
    setenv("BUILD_DEPS_TO_PB", "false", 1);
    setenv("DEBUG_REINDEX_DEPS_TO_PB", "false", 1);
    setenv("INFER_ARG_STRATEGY", "false", 1);
    setenv("INFER_DEPS_FROM_PATH", "false", 1);
  }
  return std::make_pair(pb_idx, new_strategies_reindex);
}

static int _DFS(const StrategyMap& strategy_map, const HloInstructionSequence& sequence, 
      const LeafStrategies& leaf_strategies, HloInstruction* current, 
      std::vector<ParallelBlock>& parallel_blocks, std::set<HloInstruction*>& visited) {
        if (!current || visited.count(current) > 0) {
            return -1;
        }
        visited.insert(current);
        int pb_id = InParallelBlock(current, sequence, strategy_map, leaf_strategies, parallel_blocks);
        if (pb_id > -1) {
            return pb_id;
        }
        for (auto user : current->users()) {
          int res = _DFS(strategy_map, sequence, leaf_strategies, user, parallel_blocks, visited);
          if (res != -1) {
            return res;
          }
        }

        return -1;
    }

static int find_related_pb(const StrategyMap& strategy_map, const HloInstructionSequence& sequence, 
      const LeafStrategies& leaf_strategies, HloInstruction* src_ins, 
      std::vector<ParallelBlock>& parallel_blocks){
  std::set<HloInstruction*> visited;
  int pb_id = _DFS(strategy_map, sequence, leaf_strategies, src_ins, parallel_blocks, visited);
  for (auto operand : src_ins->operands()) {
    int p = InParallelBlock(operand, sequence, strategy_map, leaf_strategies, parallel_blocks);
    if (p >= 0) {pb_id = p;}
  }
  return pb_id;
}

static void ClusterNotSharedBranch(
    const HloInstructionSequence& sequence,
    const LeafStrategies& leaf_strategies, const StrategyMap& strategy_map,
    const ClusterEnvironment& cluster_env, AutoShardingSolverOption& solver_option, 
    std::vector<ParallelBlock>& parallel_blocks,ReindexItemList& reindex_arr, ParallelBlockConfig pb_config, std::vector<int>& set_to_one) {
  // TODO: cluster ops that only dependent by one parallel_block
  // We perform this clustering for two cases:
  // 1. Input branch for parameters, rng, itoa, ... ops, there sharding strategy
  // can be inferred from its destination in a reverse way. As they only have
  // dependence to one parallel block, they should follow the strategy of this
  // parallel block, otherwise, it will introduce extra commnuications. Futher
  // more, an input parameter or random tensors have more flexibility than other
  // ops in computation graph because their resharding cost can be covered by
  // multiple stages swiching process.
  // 2. Many ops such like reshape, select, compare, broadcast,... are not set
  // followed to other ops because they can be used by multiple ops, previous
  // fusion method can not handle this case because they can't tell the
  // relations between multiple user ops. We check if these users lie in the
  // same parallel block, if so, this reshape can be clustered to any user
  // freely.
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  DepsToDotMap deps_to_dot_map;

  int _start, _end;
  const char* str_start = std::getenv("P_START");
  const char* str_end = std::getenv("P_END");
  if (str_start!=nullptr){
    _start = std::atoi(str_start);
  } else {
    _start = 0;
  }
  if (str_end!=nullptr){
    _end = std::atoi(str_end);
  } else {
    _end = instructions.size();
  }
  for (int64_t instruction_id = _start; instruction_id < _end;
  //for (int64_t instruction_id = 0; instruction_id < instructions.size();
       ++instruction_id) {
    HloInstruction* src_ins = instructions[instruction_id];
    // IMPORTANT: weried setting. try to remove in global_analysis.
    if (solver_option.mode=="global_analysis" || solver_option.mode=="search"){
      if (instruction_id == 0) set_to_one.push_back(instruction_id);
    }
    if (src_ins->users().size() == 0 || src_ins->opcode() == HloOpcode::kDot ||
        src_ins->opcode() == HloOpcode::kConvolution ||
        src_ins->opcode() == HloOpcode::kConstant) {
      continue;
    }
    // check if this inst are not set followed to other operators
    int64_t src_leaf_id = strategy_map.at(src_ins)->id;
    if (src_leaf_id == -1) {
      std::cerr << "[WARRNING]: tuple node: "
                << src_ins->ToString(HloPrintOptions::ShortParsable()) << "\n";
      continue;
    }
    if (leaf_strategies[src_leaf_id]->following != nullptr) {
      continue;
    }
    if (src_ins->shape().rank() == 0) {
      continue;
    }

    std::pair<int, std::vector<int>> P;
    P = AnalysisDependenceToPB(
        sequence, src_ins, leaf_strategies, strategy_map, cluster_env,
        parallel_blocks, solver_option, deps_to_dot_map, reindex_arr, pb_config);
    auto dimension_multiply = [](const Shape& inst_shape) -> int {
      int res = 1;
      for (int i = 0; i < inst_shape.rank(); ++i){
        res *= inst_shape.dimensions(i);
      }
      return res;
    };
    bool force_plan = false;
    for (auto p : pb_config.GetMergeInst()){
      if (p.first.second == instruction_id){
        force_plan = true;
      }
    }
    if (src_ins->users().size() != 0){
      if (src_ins->users()[0]->opcode() == HloOpcode::kTuple) {force_plan = true;}
    }
    if (P.first == -1 && (src_ins->opcode() == HloOpcode::kReshape || 
            src_ins->opcode() == HloOpcode::kSelect || 
            src_ins->opcode() == HloOpcode::kBroadcast || 
            src_ins->opcode() == HloOpcode::kAnd ||
            src_ins->opcode() == HloOpcode::kIota ||
            src_ins->opcode() == HloOpcode::kParameter) && dimension_multiply(src_ins->shape()) <= 4096*1024 && !force_plan) {
      bool exist_in_multi_pb_infer_list = false;
      for (int idx = 0; idx < reindex_arr.reindex_items().size(); ++idx) {
        ReindexItem item = reindex_arr.reindex_items(idx);
        if (item.inst_id() == instruction_id){
          exist_in_multi_pb_infer_list = true;
        }
      }
      if (!exist_in_multi_pb_infer_list){
        set_to_one.push_back(instruction_id);
        ///std::cerr << "Find a inst that are small and not in multi_pb_infer_list." << instruction_id << "\n";
      }
    }
    bool multi_pb = false;
    for (int idx = 0; idx < reindex_arr.reindex_items().size(); ++idx) {
        ReindexItem item = reindex_arr.reindex_items(idx);
        if (item.inst_id() == instruction_id){
          multi_pb = true;
        }
      }
    if (P.first == -1 && !multi_pb && force_plan && (solver_option.mode == "analysis_segment" || solver_option.mode == "profile" || solver_option.mode == "profile_segment" )){
      set_to_one.push_back(instruction_id);
      //std::cerr << "\n\nFind a inst in segment that can not be handled, force set to R." << instruction_id << "\n\n";
    }
    if (P.first == -1) {
      continue;
    }

    // add to parallel block
    parallel_blocks[P.first].first.push_back(src_ins);
    parallel_blocks[P.first].second.push_back(P.second);
  }
  return;
}

template <typename T>
void keepFirstN(std::vector<T>& vec, size_t n) {
  if (n < 0 || n > vec.size()) {
    return;
  }
  vec.erase(vec.begin() + n, vec.end());
}

static void ResetInstParallelPlan(const HloInstructionSequence& sequence,
                                 const StrategyMap& strategy_map, const DotPairs& new_plan,
                                 std::vector<ParallelBlock>& parallel_blocks){
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (auto pair : new_plan) {
    int inst_id = pair.first.second;
    for (int pb_id = 0; pb_id < parallel_blocks.size(); ++pb_id) {
      for (int idx = 0; idx < parallel_blocks[pb_id].first.size(); ++idx) {
        if (strategy_map.at(parallel_blocks[pb_id].first[idx])->instruction_id == inst_id) {
          //std::cerr << "Reset inst plan for " << "inst_id: " << inst_id << " ...";
          parallel_blocks[pb_id].second[idx] = pair.second;
        }
      }
    }
  }
}

static void build_new_parallel_plan(std::vector<ParallelBlock>& parallel_blocks, 
                      AutoShardingSolverOption& solver_option, DotPairs& new_pb_plan) {

    if(parallel_blocks.size()/solver_option.num_layers != 9 || solver_option.bs_size >= 80){
      return;
    }
    bool has_bmm = false;
    for (const auto& pb : parallel_blocks) {
      if (!pb.second.empty() && pb.second[0].size() == 4) {
              has_bmm = true;
              break; 
        }
    }
    if (has_bmm) {return;}
    else{ 
      std::string new_plan_str = "0 2 1 1 0 , 1 2 1 1 0, 3 2 1 1 0, 5 2 2 1 1";
      for (int i = 0; i < solver_option.num_layers; ++i){
        parseDotPairs(new_plan_str, new_pb_plan, i*9, 0);
      }
    }
    return;
}

static void ResetDotParallelPlan(const HloInstructionSequence& sequence,
                                 const DotPairs& rm, const DotPairs& add, const DotPairs& new_plan,
                                 std::vector<ParallelBlock>& parallel_blocks, AutoShardingSolverOption& solver_option) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  DotPairs new_pb_plan;
  if (new_plan.size() == 0){
    build_new_parallel_plan(parallel_blocks, solver_option, new_pb_plan);
  }else{
    new_pb_plan = new_plan;
  }
  for (auto pair : new_pb_plan) {
    int pb_id = pair.first.first;
    int id_in_pb = pair.first.second;
    parallel_blocks[pb_id].second[id_in_pb] = pair.second;
    // std::cerr << "reset pb_id " << pb_id << " idx " << id_in_pb << "\n";
  }
  for (auto pair : rm) {
    int pb_id = pair.first.first;
    int id_in_pb = pair.first.second;
    std::cerr << "remove " << id_in_pb << " in pb " << pb_id << "\n";
    parallel_blocks[pb_id].first.erase(parallel_blocks[pb_id].first.begin() +
                                       id_in_pb);
    parallel_blocks[pb_id].second.erase(parallel_blocks[pb_id].second.begin() +
                                        id_in_pb);
  }
  for (auto pair : add) {
    int inst_id = pair.first.first;
    int pb_id = pair.first.second;
    std::cerr << "add inst " << inst_id << " to pb " << pb_id << "\n";
    parallel_blocks[pb_id].first.push_back(instructions[inst_id]);
    parallel_blocks[pb_id].second.push_back(pair.second);
  }
}

static int64_t GetLossPosition(const HloInstructionSequence& sequence, const ParallelBlockConfig& pb_config){
  CHECK_GE(pb_config._forward_dot_count, 1);
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  int dot_count = 0;
  int64_t instruction_id = 0;
  for (;instruction_id < instructions.size(); ++instruction_id) { 
      if(instructions[instruction_id]->opcode() == HloOpcode::kDot){
        dot_count++;
      }
      if(dot_count == pb_config._forward_dot_count){
        break;
      }         
  }
  
  return ++instruction_id;
}

// void ClusterInstWithMultiPBUser(const HloInstructionSequence& sequence, 
//                          const LeafStrategies& leaf_strategies,
//                          const StrategyMap& strategy_map,
//                          const ClusterEnvironment& cluster_env,
//                          const InstructionDepthMap& depth_map,
//                          std::vector<ParallelBlock>& parallel_blocks) {
//   // all instructions that not in one parallel block.
//   for (int64_t instruction_id = 0; instruction_id < instructions.size();
//        ++instruction_id) {
//     HloInstruction* src_ins = instructions[instruction_id];
//     if (src_ins->users().size() == 0 || src_ins->opcode() == HloOpcode::kDot ||
//         src_ins->opcode() == HloOpcode::kConvolution ||
//         src_ins->opcode() == HloOpcode::kConstant) {
//       continue;
//     }
//     // check if this inst are not set followed to other operators
//     int64_t src_leaf_id = strategy_map.at(src_ins)->id;
//     if (src_leaf_id == -1) {
//       std::cerr << "[WARRNING]: tuple node: "
//                 << src_ins->ToString(HloPrintOptions::ShortParsable()) << "\n";
//       continue;
//     }
//     if (leaf_strategies[src_leaf_id]->following != nullptr) {
//       continue;
//     }
//     if (InParallelBlock(src_ins, sequence, strategy_map, leaf_strategies,
//                                 parallel_blocks)!=-1){
//       continue;
//     }
//     std::cerr << "instruction id " << instruction_id << "\n";
//     // handle different situations.
//     // 1. Argument for multi-layesr
//     // 2. Argument for different PB in one layer
//     // 3. Reshape and Rng 
//     // 4. Broadcast for multi-layers
//     // 5. Compare for multi parallel bloock

//   }
// }

void BuildParallelBlocks(const HloInstructionSequence& sequence,
                         const LeafStrategies& leaf_strategies,
                         const StrategyMap& strategy_map,
                         const ClusterEnvironment& cluster_env,
                         const ParallelBlockConfig& pb_config,
                         std::vector<ParallelBlock>& parallel_blocks,
                         ReindexItemList& reindex_arr,
                         AutoShardingSolverOption& solver_option, std::vector<int>& set_to_one) {
  int64_t loss_pos = GetLossPosition(sequence, pb_config);
  const bool dump_pb_ = std::getenv("DEBUG_PB_GENERATION");
  if (dump_pb_){
    std::cerr << "Loss Position is " << loss_pos;
  }
  // find all forward dot/convs and its backward op, generate one-to-one
  // strategy mapping relations for each pair.
  if (dump_pb_){
    std::cerr << "BuildPB: ClusterForwardBackwardDot";
  }
  ClusterForwardBackwardDot(sequence, leaf_strategies, strategy_map,
                            cluster_env, parallel_blocks, loss_pos);
  if (dump_pb_){
    std::cerr << "PB group forward and backward \n" << DumpParallelBlocks(parallel_blocks, strategy_map);
  }
  ResetDotParallelPlan(sequence, pb_config.GetRemoveDotPlan(), pb_config.GetAddDotPlan(), pb_config.GetNewDotPlan(), 
                       parallel_blocks, solver_option);
  if (dump_pb_){
    std::cerr << "BuildPB: ClusterAdjacentDot";
  }
  ClusterAdjacentDot(sequence, leaf_strategies, strategy_map, cluster_env,
                    pb_config.GetDotPairs(), parallel_blocks, solver_option);
  if (dump_pb_){
    std::cerr << "BuildPB: ClusterNotSharedBranch";
  }
  ClusterNotSharedBranch(sequence, leaf_strategies, strategy_map, cluster_env,
                         solver_option, parallel_blocks, reindex_arr, pb_config, set_to_one);
  if (dump_pb_){
    std::cerr << "PB after CluterNotSharedBranch\n" << DumpParallelBlocks(parallel_blocks, strategy_map);
  }
  // Force merge inst to current parallel block if specificated.
  for (auto pair : pb_config.GetMergeInst()) {
    MergeInstToParallelBlocks(sequence, parallel_blocks, pair.first.first,
                              pair.first.second, pair.second);
  }
  ResetInstParallelPlan(sequence, strategy_map, pb_config.GetNewInstPlan(), parallel_blocks);

  // simply set -1 in parallel plans to 0.
  for (auto& pb : parallel_blocks) {
    for (auto& plan : pb.second) {
        std::transform(plan.begin(), plan.end(), plan.begin(),
                       [](int p) { return p == -1 ? 0 : p; });
    }
  }
  
  if (solver_option.mode == "analysis_global"){
    int dump_env = 1;
    const char* dump_env_str = std::getenv("DUMP_PROFILE_PARALLEL_BLOCK");
    if (dump_env_str != nullptr) {
        dump_env = std::atoi(dump_env_str);
    }
    if(dump_env!=0){
      std::cerr << "################## FUSED PARALLEL_BLOCKS ################### \n" 
        << DumpParallelBlocks(parallel_blocks, strategy_map) << "\n #########################################";
    }
  }
  
  std::vector<ParallelBlock> tmp_parallel_blocks;
  for (int num = 0; num < parallel_blocks.size(); ++num) {
    ParallelBlock pb = parallel_blocks[num];
    tmp_parallel_blocks.push_back(pb);
  }
  parallel_blocks.clear();
  int pb_num_i = 0;
  const char* pb_num = std::getenv("PB_NUM");
  if (pb_num != nullptr) {
    pb_num_i = std::atoi(pb_num);
  } else {
    pb_num_i = tmp_parallel_blocks.size();
  }
  for (int num = 0; num < pb_num_i; ++num) {
    parallel_blocks.push_back(tmp_parallel_blocks[num]);
  }

  if (pb_num != nullptr) {
    LOG(INFO)<<DumpParallelBlocks(parallel_blocks, strategy_map);
  }
}

// reset strategies for all parallel block instructions in the sequence, as
// inst's resharding cost vector depends on its operands strategies, we also
// reset the resharding cost for each instruction.
void ResetStrategiesForParallelBlocks(
    const HloInstructionSequence& sequence,
    const std::vector<ParallelBlock>& parallel_blocks,
    LeafStrategies& leaf_strategies, StrategyMap& strategy_map, std::vector<int>& set_to_one, AutoShardingSolverOption& solver_option) {

  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  int _start, _end;
  const char* str_start = std::getenv("R_START");
  const char* str_end = std::getenv("R_END");
  if (str_start!=nullptr){
    _start = std::atoi(str_start);
  } else {
    _start = 0;
  }
  if (str_end!=nullptr){
    _end = std::atoi(str_end);
  } else {
    _end = instructions.size();
  }

  for (size_t instruction_id = _start; instruction_id < _end;
       ++instruction_id) {
    const auto ins = instructions[instruction_id];
    auto block_idx = GetParallelBlock(ins, parallel_blocks);
    // reset leaf_vector for ins strategies if it belongs to a parllel block
    if (block_idx.first != -1) {
      const ParallelBlock* parallel_block = &parallel_blocks[block_idx.first];
      LOG(INFO) << "\nReset [Stras] for : \n  "
                << strategy_map.at(ins)->instruction_id << " "
                << ins->ToString(HloPrintOptions::ShortParsable())
                << "\n     reset cost vector for related users: ";
      std::vector<ShardingStrategy>* old_leaf_vector =
          &strategy_map.at(ins)->leaf_vector;
      std::vector<ShardingStrategy> new_leaf_vector;
      // set leaf vector from old leaf vector and parallel plan.
      for (const auto strategy_idx : parallel_block->second[block_idx.second]) {
        // std::cerr << "strategy_idx: " << strategy_idx << "\n";
        // create new sharding strategy from old one.
        new_leaf_vector.push_back(old_leaf_vector->at(strategy_idx));
      }
      strategy_map.at(ins)->leaf_vector = std::move(new_leaf_vector);
    }
    // set 
    auto it = std::find(set_to_one.begin(), set_to_one.end(), instruction_id);
    if (it != set_to_one.end()){
      //std::cerr << "############RESET INSTRUCTION " << instruction_id << "\n";
      std::vector<ShardingStrategy>* old_leaf_vector =
          &strategy_map.at(ins)->leaf_vector;
      std::vector<ShardingStrategy> new_leaf_vector;
      for (int idx = 0; idx < old_leaf_vector->size(); ++idx){
        if (old_leaf_vector->at(idx).name == "R"){
          new_leaf_vector.push_back(old_leaf_vector->at(idx));
        }
      }
      strategy_map.at(ins)->leaf_vector = std::move(new_leaf_vector); 
    }
    // reset resharding cost for each instruction if its operand belongs to a
    // parallel block.
    if ((solver_option.mode == "analysis_segment" || solver_option.mode == "profile_segment") 
              && ins->opcode() == HloOpcode::kTuple){
      for (int operand_idx = 0; operand_idx < ins->operand_count();
                ++operand_idx) {
        auto it = std::find(set_to_one.begin(), set_to_one.end(), strategy_map.at(ins->operand(operand_idx))->instruction_id);
        if (it != set_to_one.end()){
          for (auto& strategy : strategy_map.at(ins)->childs[operand_idx]->leaf_vector) {
                   std::vector<double> resharding_vec;
                   resharding_vec.push_back(strategy.name == "R" ? 0 : 1e+13);
                   strategy.resharding_costs[0].clear();  // TODO: check if this is necessary
                   strategy.resharding_costs[0] = std::move(resharding_vec);
          }
          bool have_zero = false;
          for (auto& strategy : strategy_map.at(ins)->childs[operand_idx]->leaf_vector) {
            if (strategy.resharding_costs[0][0] == 0){
                     have_zero = true;
            }
          }
          if (!have_zero) {LOG(FATAL) << "set to one have some problem.";}
        }
        auto operand_block_id =
                 GetParallelBlock(ins->operand(operand_idx), parallel_blocks);
        if (operand_block_id.first == -1) continue;
        LOG(INFO) << strategy_map.at(ins)->instruction_id << ", ";
        const ParallelBlock* operand_parallel_block =
                 &parallel_blocks[operand_block_id.first];
        for (auto& strategy : strategy_map.at(ins)->childs[operand_idx]->leaf_vector) {
                  std::vector<double> resharding_vec;
          for (const int operand_strategy_idx :
                       operand_parallel_block->second[operand_block_id.second]) {
                    resharding_vec.push_back(
                        strategy.resharding_costs[0][operand_strategy_idx]);
          }
          strategy.resharding_costs[0]
                      .clear();  // TODO: check if this is necessary
          strategy.resharding_costs[0] = std::move(resharding_vec);
        }
      }
      continue;
    }
    for (int operand_idx = 0; operand_idx < ins->operand_count();
         ++operand_idx) {
      auto it = std::find(set_to_one.begin(), set_to_one.end(), strategy_map.at(ins->operand(operand_idx))->instruction_id);
      if (it != set_to_one.end()){
          for (auto& strategy : strategy_map.at(ins)->leaf_vector) {
            std::vector<double> resharding_vec;
            resharding_vec.push_back(strategy.name == "R" ? 0 : 1e+13);
            // for (int operand_strategy_idx =0; 
            //   operand_strategy_idx < strategy_map.at(ins->operand(operand_idx))->leaf_vector.size(); 
            //   ++operand_strategy_idx){
            //     if (strategy_map.at(ins->operand(operand_idx))->leaf_vector[operand_strategy_idx].name == "R"){
            //       std::cerr << "    operand_strategy_idx " << operand_strategy_idx << "\n";
            //       resharding_vec.push_back(strategy.resharding_costs[operand_idx][operand_strategy_idx]);
            //     }
            // }
            strategy.resharding_costs[operand_idx].clear();  // TODO: check if this is necessary
            strategy.resharding_costs[operand_idx] = std::move(resharding_vec);
          }
          bool have_zero = false;
          for (auto& strategy : strategy_map.at(ins)->leaf_vector) {
            if (strategy.resharding_costs[operand_idx][0] == 0){
              have_zero = true;
            }
          }
          if (!have_zero) {LOG(FATAL) << "set to one have some problem.";}
      }
      auto operand_block_id =
          GetParallelBlock(ins->operand(operand_idx), parallel_blocks);
      if (operand_block_id.first == -1) continue;
      LOG(INFO) << strategy_map.at(ins)->instruction_id << ", ";
      const ParallelBlock* operand_parallel_block =
          &parallel_blocks[operand_block_id.first];
      for (auto& strategy : strategy_map.at(ins)->leaf_vector) {
        std::vector<double> resharding_vec;
        for (const int operand_strategy_idx :
             operand_parallel_block->second[operand_block_id.second]) {
          resharding_vec.push_back(
              strategy.resharding_costs[operand_idx][operand_strategy_idx]);
        }
        strategy.resharding_costs[operand_idx]
            .clear();  // TODO: check if this is necessary
        strategy.resharding_costs[operand_idx] = std::move(resharding_vec);
      }
    }

    // check resharding cost length for each instruction
    for (int operand_idx = 0; operand_idx < ins->operand_count();
         ++operand_idx) {
      for (auto& strategy : strategy_map.at(ins)->leaf_vector) {
        int resharding_cost_length =
            strategy.resharding_costs[operand_idx].size();
        int op_leaf_vector_size = 0;
        if (strategy_map.at(ins->operand(operand_idx))->is_tuple) {
          op_leaf_vector_size = strategy_map.at(ins->operand(operand_idx))
                                    ->childs[0]
                                    ->leaf_vector.size();
        } else {
          op_leaf_vector_size =
              strategy_map.at(ins->operand(operand_idx))->leaf_vector.size();
        }
        CHECK_EQ(resharding_cost_length, op_leaf_vector_size)
            << " ERROR: " << instruction_id << " "
            << ins->ToString(HloPrintOptions::ShortParsable())
            << "  operand_id: " << operand_idx << "\n";
      }
    }
  }
  // Do not set following for instructions in parallel block at here, we set
  // cost graph edge merge pair after cost graph simplification.
}

void CostGraph::RuleBasedMergeNodeForPBdst(
    int src, int dst, const std::vector<int>& dst_parallel_plan,
    int dst_origin_stra_num) {
  CHECK(adjacency[src].count(dst));
  CHECK(adjacency[dst].count(src));
  CHECK(!merged_to_.count(src));
  CHECK(!merged_to_.count(dst));
  CHECK_NE(src, dst);
  CHECK_EQ(dst_parallel_plan.size(), node_lens[dst]);

  Matrix edge_cost = GetEdgeCost(dst, src);

  std::vector<int> reindexing(node_lens[dst]);
  if (dst_origin_stra_num == node_lens[src]) {
    reindexing.assign(dst_parallel_plan.begin(), dst_parallel_plan.end());
  } else {
    // Otherwise, find the strategy to follow greedily.
    // For every straetgy in dst, find the strategy in src with
    // the lowest resharding cost.
    std::vector<int> arange(node_lens[src]);
    std::iota(arange.begin(), arange.end(), 0);
    for (int i = 0; i < node_lens[dst]; ++i) {
      std::vector<std::pair<double, int>> keys;
      // If there are multiple strategies with the same lowest costs,
      // prefer to follow "replicated", which has the largest index.
      // Node: We assume the strategy "Repilcated" is always appended
      // as the last strategy in BuildStrategyAndCost.
      // [TODO]: This may need to be modified because the order of the
      // strategies we regenerate may not be the same as before, so we cannot
      // use the strategy number to determine the weight when multiple
      // strategies have the same minimum cost.。
      for (int j = 0; j < node_lens[src]; ++j) {
        keys.push_back({edge_cost(i, j), -j});
      }

      std::sort(arange.begin(), arange.end(), [&keys](int l, int r) {
        return (keys[l].first < keys[r].first) ||
               (keys[l].first == keys[r].first &&
                keys[l].second < keys[r].second);
      });

      reindexing[i] = arange.front();
    }
  }
  merged_to_[src] = dst;
  reindexing_vector[src] = reindexing;

  // Merge edge cost matrix
  std::vector<int> adj_list(adjacency[src].begin(), adjacency[src].end());
  for (int adj : adj_list) {
    if (adj == dst) {
      for (int i = 0; i < node_lens[dst]; ++i) {
        extra_node_costs[dst][i] += edge_cost(i, reindexing[i]);
      }
    } else {
      Matrix added_edge_cost(node_lens[dst], node_lens[adj]);
      Matrix edge_cost_src_adj = GetEdgeCost(src, adj);

      for (int i = 0; i < node_lens[dst]; ++i) {
        for (int k = 0; k < node_lens[adj]; ++k) {
          added_edge_cost(i, k) = edge_cost_src_adj(reindexing[i], k);
        }
      }

      AddEdgeCost(dst, adj, added_edge_cost);
    }
  }

  // Remove edges
  for (int adj : adj_list) {
    RemoveEdge(src, adj);
  }
}

void CostGraph::RuleBasedSimplifyAfterStrategiesReorder(
    const HloInstructionSequence& sequence,
    const LeafStrategies& leaf_strategies,
    const std::vector<ParallelBlock>& parallel_blocks,
    std::vector<int>& origin_stra_num) {
  for (const auto& pair : to_merge_pairs_) {
    int src = pair.first;
    int dst = pair.second;
    dst = QueryDestination(dst);
    // Merge src nodes that has been set following to dst, typically the
    // strategies reindexing should be 0->1->...->node.len() because the
    // strategies of two node are generated in a same way, but we may have
    // change strategies order for some inst in parallel blocks, we thus merge
    // following pairs by using a new reindexing vector generated by parallel
    // plans.
    const std::vector<HloInstruction*>& instructions = sequence.instructions();
    int dst_ins_id = leaf_strategies[dst]->instruction_id;
    std::pair<int, int> dst_pb_idx =
        GetParallelBlock(instructions[dst_ins_id], parallel_blocks);
    if (dst_pb_idx.first == -1) {
      MergeNode(src, dst);
    } else {
      // get dst parallel plans
      const ParallelBlock* dst_pb = &parallel_blocks[dst_pb_idx.first];
      const std::vector<int> dst_parallel_plan =
          dst_pb->second[dst_pb_idx.second];
      int dst_origin_stra_num = origin_stra_num[dst];
      RuleBasedMergeNodeForPBdst(src, dst, dst_parallel_plan,
                                 dst_origin_stra_num);
    }
  }
  // do not build following idx anymore.
}

void CostGraph::MergeNodeForParallelBlock(int src, int dst) {
  CHECK(adjacency[src].count(dst));
  CHECK(adjacency[dst].count(src));
  CHECK(!merged_to_.count(src));
  CHECK(!merged_to_.count(dst));
  CHECK_NE(src, dst);

  Matrix edge_cost = GetEdgeCost(dst, src);

  std::vector<int> reindexing(node_lens[dst]);
  std::iota(reindexing.begin(), reindexing.end(), 0);
  merged_to_[src] = dst;
  reindexing_vector[src] = reindexing;

  // Merge edge cost matrix
  std::vector<int> adj_list(adjacency[src].begin(), adjacency[src].end());
  for (int adj : adj_list) {
    if (adj == dst) {
      for (int i = 0; i < node_lens[dst]; ++i) {
        extra_node_costs[dst][i] += edge_cost(i, reindexing[i]);
      }
    } else {
      Matrix added_edge_cost(node_lens[dst], node_lens[adj]);
      Matrix edge_cost_src_adj = GetEdgeCost(src, adj);

      for (int i = 0; i < node_lens[dst]; ++i) {
        for (int k = 0; k < node_lens[adj]; ++k) {
          added_edge_cost(i, k) = edge_cost_src_adj(reindexing[i], k);
        }
      }
      AddEdgeCost(dst, adj, added_edge_cost);
    }
  }
  // Remove edges
  for (int adj : adj_list) {
    RemoveEdge(src, adj);
  }
}

void CostGraph::SimplifyParallelBlockInCostGraph(
    std::vector<std::pair<int, int>>& merge_pairs) {
  bool changed = true;
  LOG(INFO) << "Start CostGraph::SimplifyParallelBlockInCostGraph.";
  LOG(INFO) << "Merge pairs size before: " << merge_pairs.size() << "\n";
  std::vector<std::pair<int, int>> merged_for_dump;
  while (!merge_pairs.empty() && changed) {
    changed = false;
    // try to merge all nodes in parallel block
    for (auto& pair : merge_pairs) {
      int src = pair.first;
      int dst = pair.second;
      // skip if there are no cost edges between src and dst currently.
      if (!adjacency[src].count(dst) || !adjacency[dst].count(src)) {
        continue;
      }
      LOG(INFO) << "Merging "<<src << "->" << dst << " ";
      MergeNodeForParallelBlock(src, dst);
      // remove pair from merge_pairs
      merge_pairs.erase(std::remove(merge_pairs.begin(), merge_pairs.end(),
                                    std::make_pair(src, dst)),
                        merge_pairs.end());

      merged_for_dump.push_back(std::make_pair(src, dst));
      changed = true;
    }
  }
  LOG(INFO) << "Merge pairs size after: " << merge_pairs.size() ;
  std::sort(merged_for_dump.begin(), merged_for_dump.end(), 
    [](const std::pair<int, int>& left, std::pair<int, int>& right) -> bool {return left.first < right.first;}
    );
  // LOG(INFO) << "Merge For Dump: " ;
  // for (auto pair : merged_for_dump) {
  //   LOG(INFO) << pair.first << "->" << pair.second << ", ";
  // }
  follow_idx.clear();  // rebuild follow map
  follow_idx.reserve(node_lens.size());
  for (int i = 0; i < node_lens.size(); ++i) {
    if (merged_to_.count(i)) {
      follow_idx.push_back(QueryDestination(i));
    } else {
      follow_idx.push_back(-1);
    }
  }
  }

void ReduceEdgesForParallelBlocks(
    const HloInstructionSequence& sequence,
    const LeafStrategies& leaf_strategies, const StrategyMap& strategy_map,
    CostGraph& cost_graph, const std::vector<ParallelBlock>& parallel_blocks,
    std::vector<int>& failed_merged) {
  LOG(INFO) << "Start ReduceEdgesForParallelBlocks...";
  std::vector<std::pair<int, int>> merge_pairs;
  // build a merge_pairs which will merge all nodes in parallel block to
  // one, and the merge need to be followed by some order.
  for (auto& parallel_block : parallel_blocks) {
    if (parallel_block.first.size() < 1) {
      continue;
    }
    int dst_leaf_id = strategy_map.at(parallel_block.first[0])->id;
    // for instructions in parallel block except dst, set their following to
    // dst.
    for (int i = 1; i < parallel_block.first.size(); ++i) {
      HloInstruction* ins = parallel_block.first[i];
      strategy_map.at(ins)->following = leaf_strategies[dst_leaf_id];
      merge_pairs.push_back(
          std::make_pair(strategy_map.at(ins)->id, dst_leaf_id));
    }
  }
  //  reset cost graph and follow_idx
  LOG(INFO) << "Reset cost graph and follow_idx: cost_graph.SimplifyParallelBlockInCostGraph(merge_pairs)";
  cost_graph.SimplifyParallelBlockInCostGraph(merge_pairs);
  for (auto pair : merge_pairs) {
    int src = pair.first;
    failed_merged.push_back(src);
  }
}

static int64_t pick_stra_satisify_all(std::vector<ShardingStrategy>& needed, const StrategyVector* stras) {
  std::vector<std::vector<int64_t>> compatible_indexes;
  // std::cerr << "needed size: " << needed.size() << " stras size: " << stras->leaf_vector.size() << "\n";
  
  for (auto& shard_stra : needed) {
    // std::cerr << "    needed tile assign is : ";
    auto needed_tile_assign = shard_stra.output_sharding.tile_assignment().dimensions();
    // for (auto a : needed_tile_assign) {std::cerr << a << " ";}
    // std::cerr << "\n";
    std::vector<int64_t> compatible_index;
    // check compatiblity with all tile_assign in stras
    for (int i = 0; i < stras->leaf_vector.size(); ++i){
      auto stra = stras->leaf_vector[i];
      auto current_tile_assign = stra.output_sharding.tile_assignment().dimensions();
      // std::cerr<< "    ----current_tile_assign: ";
      //     for (auto a : current_tile_assign) {std::cerr << a << " ";} 
      // std::cerr << "\n";
      if (!stra.output_sharding.IsReplicated()){
        // check each elem in tile_assign of stra < elem in tile_assign of every shard_stra
        bool compatible = true;
        for (int j = 0; j < std::min(needed_tile_assign.size(), current_tile_assign.size()); ++j){
          if (needed_tile_assign[j] < current_tile_assign[j]){
            compatible = false;
            break;
          }
          if ((current_tile_assign[j] > 1) && 
            (check_device_order(stra) != check_device_order(shard_stra))){
            // check device dimension 
            compatible = false;
            break;
          }
        }
        if (shard_stra.output_sharding.IsReplicated()){
          compatible = false;
        }
        if(!compatible){
          continue;
        }
      }
      compatible_index.push_back(i);
    }
    compatible_indexes.push_back(compatible_index);
  }
    // find the first stra that is compatible with all shard_stra
    // find the commen element in all compatible_index in compatible_indexes
    std::vector<int64_t> common_index;
    for (int i = 0; i < compatible_indexes.size(); ++i){
      if (i == 0){
        common_index = compatible_indexes[i];
      }else{
        std::vector<int64_t> tmp;
        std::set_intersection(common_index.begin(), common_index.end(), compatible_indexes[i].begin(), compatible_indexes[i].end(), std::back_inserter(tmp));
        common_index = tmp;
      }
    }
    if (common_index.empty()){
      return -1;
    }
    return common_index[0];
}

// TODO
void InferSolutionForRemainInsts(const HloInstructionSequence& sequence, 
    const LeafStrategies& leaf_strategies, const StrategyMap& strategy_map, const CostGraph& cost_graph,
    const std::vector<ParallelBlock>& parallel_blocks, std::vector<long int>& solution_vector, const std::string reindex_path){
    // load reindexing map for insts used by multiple parallel blocks.
    int generated_sval = 0;
    ReindexItemList reindex_arr;
    LoadReindexFromFile(reindex_arr, reindex_path);

    const std::vector<HloInstruction*>& instructions = sequence.instructions();
    std::vector<long int> solution_vector_infered(reindex_arr.reindex_items().size(), -1);
    int key_inst_count = 0;
    for (auto item : reindex_arr.reindex_items()){
      if (item.is_key_inst()){
        int64_t leaf_id = item.inst_leaf_id();
        int64_t* v = &solution_vector_infered[leaf_id];
        *v = solution_vector[key_inst_count++];
      }
    }
    CHECK_EQ(key_inst_count, solution_vector.size()) << " Solution Vector Size Not Match.";
    int last = -1;
    for (auto item : reindex_arr.reindex_items()){

      generated_sval += 1;
      const HloInstruction* inst = instructions[item.inst_id()];
      int64_t leaf_id = item.inst_leaf_id();
      if (leaf_id == last) {
        std::cerr << leaf_id << " REPLICATED!\n";
      }
      last = leaf_id;
      int64_t* v = &solution_vector_infered[leaf_id];
      if (item.is_key_inst()){
        continue; // first dot/conv in parallel block, strategy has been choosed in enumeration.
      }
      if (item.stra_num() == 1){
        *v = 0; // strategy_vector has only one strategy.
        continue;
      }
      if (!item.deps_by_multi_pb()){
        // keep same strategy index with followed dot/conv or other insts(like args deps by two pb), 
        // this solution value will be reindexed later in SetHloSharding.
        *v = solution_vector_infered[item.deps_pb_id(0)]; 
        continue;
      } else { // insts depent by multiple parallel block, infer its strategy to satisify all dependent dot/conv.
        const StrategyVector* stras = leaf_strategies[leaf_id];
        std::vector<ShardingStrategy> needed_stras;
        for (int idx = 0; idx < item.deps_pb_id_size(); ++idx){
          int deps_stra_idx = solution_vector_infered[item.deps_pb_id(idx)];
          ReindexData reindex_data = item.reindexing(idx);
          needed_stras.push_back(stras->leaf_vector[reindex_data.reindex(deps_stra_idx)]);
        }
        int64_t value = pick_stra_satisify_all(needed_stras, stras);

        // [TODO:huwf][FIXME]: for arg_4.5, set R in every strategies, its idx are different in llama7b and llama70b
        if (leaf_id == 121 || leaf_id == 123) {
          HloInstruction* arg_4_5 = instructions[leaf_strategies[leaf_id]->instruction_id];
          if (arg_4_5->opcode() == HloOpcode::kParameter){
            value = leaf_id == 121 ? 3 : 4;
          }
        }
        // std::cerr << "pick stra number: " << value << "\n";
        if (value != -1){
          *v = value;
          continue;
        } else {
          LOG(FATAL) << "Cannot Infer Solution Value For Inst " << item.inst_id() << " Deps By Multiple Parallel Blocks.";
        }
      }
      LOG(FATAL) << "Cannot Infer Solution Value For Inst " << item.inst_id();
    }
    CHECK_EQ(generated_sval, cost_graph.node_lens.size()) 
      << " Solution Vector Size Not Match. generated_sval " << generated_sval 
      << " nodes " << cost_graph.node_lens.size();
    solution_vector = std::move(solution_vector_infered);
  //   for (auto i : solution_vector) {std::cerr << i << " ";}
  //  std::cerr << "DONE\n";
}


static bool special_handle_for_moe(AutoShardingSolverOption& solver_option, const std::vector<ParallelBlock>& parallel_blocks, 
        std::vector<std::pair<int, int>>& interval_list){
  bool has_bmm = false;
  for (const auto& pb : parallel_blocks) {
    if (!pb.second.empty() && pb.second[0].size() == 4) {
        has_bmm = true;
        break; 
    }
  }
  if (!has_bmm){
    return false;
  }
  else {
    for (int i = 0; i < solver_option.num_layers; ++i){
      interval_list.push_back(std::make_pair(0, 8));
      interval_list.push_back(std::make_pair(8, 14));
    }
    return true;
  }

}

static std::pair<int, int> get_pb_seq_origin_idx(std::vector<std::pair<int, int>>& interval_list, int interval_idx){
  int start_idx = 0;
  for (int i=0; i<interval_idx; ++i){
    start_idx += (interval_list[i].second-interval_list[i].first);
  }
  int end_idx = start_idx + (interval_list[interval_idx].second-interval_list[interval_idx].first);
  return std::make_pair(start_idx, end_idx);
}

// Perform template matching on a serial PB sequence to find a set of unique segments. 
// The goal is to minimize the number of segment types while keeping the length of each segment as short as possible.
void ExtractPBSequences(const HloInstructionSequence& sequence,  const ParallelBlockConfig& pb_config, 
        const InstructionDepthMap& depth_map, AutoShardingSolverOption& solver_option, const std::vector<ParallelBlock>& parallel_blocks, 
        std::vector<int> pbs_idx, std::vector<std::pair<int, int>>& interval_list){
  if(!special_handle_for_moe(solver_option, parallel_blocks, interval_list)){
    // build all possible model segment with 3-5 pbs.
    std::vector<ParallelBlockSeq> segments;
    std::vector<std::vector<int>> interval_table(parallel_blocks.size(), 
                            std::vector<int>(parallel_blocks.size()+1, -1));
    // build interval table
    for (int seg_length = 3; seg_length < 6; ++seg_length) {
      for (int start_pb_idx = 0; start_pb_idx < (parallel_blocks.size()-seg_length + 1); ++start_pb_idx){
        ParallelBlockSeq pb_seq(sequence, pb_config, depth_map, parallel_blocks, start_pb_idx, seg_length); 
        // check if matched with existing segment
        bool have_matched = false;
        for (int seg_idx = 0; seg_idx < segments.size(); seg_idx++){
          auto seg = segments[seg_idx];
          if (seg.getSeqLength() != pb_seq.getSeqLength()){
            continue;
          }
          if (seg.match(pb_seq)){
            interval_table[start_pb_idx][start_pb_idx+seg_length] = seg_idx;
            have_matched = true;
            break;
          }
        }
        
        if (!have_matched){
          segments.push_back(pb_seq);
          interval_table[start_pb_idx][start_pb_idx+seg_length] = segments.size()-1;
        }
      }
    }

    // DP to find a minimal set of subsequence to cover whole sequence.
    std::vector<int> dp(parallel_blocks.size()+1, 99); 
    dp[0] = 0;
    std::unordered_map<int, std::vector<int>> used_seg_idxs; 
    for (int i = 0; i < parallel_blocks.size()+1; ++i){
      for (int j = i + 1; j < parallel_blocks.size()+1; ++j) {
        int mapped_seg_idx = interval_table[i][j];
        if (mapped_seg_idx == -1) {continue;}
        if (std::find(used_seg_idxs[i].begin(), used_seg_idxs[i].end(), mapped_seg_idx) != used_seg_idxs[i].end()) {
          if (dp[i] < dp[j]){ 
            dp[j] = dp[i];
            used_seg_idxs[j] = used_seg_idxs[i];
            used_seg_idxs[j].push_back(mapped_seg_idx);
          }
        }
        else{
          if(dp[i] + 1 < dp[j]){
            dp[j] = dp[i] + 1;
            used_seg_idxs[j] = used_seg_idxs[i];
            used_seg_idxs[j].push_back(mapped_seg_idx);
          }
        }
      }
    }

    // save to interval_list
    for (const auto seg_idx : used_seg_idxs[parallel_blocks.size()-1]){
      int first = segments[seg_idx].getStartIdx();
      int second = segments[seg_idx].getStartIdx() + segments[seg_idx].getSeqLength();
      interval_list.push_back(std::make_pair(first, second));
    }
 
    for (size_t i = 0; i < interval_list.size(); ++i) {
      interval_list[i].first = pbs_idx[interval_list[i].first];
      interval_list[i].second = pbs_idx[interval_list[i].second];
    }
  }else{
    interval_list.clear();
    special_handle_for_moe(solver_option, parallel_blocks, interval_list);
  }
  
  std::pair<int, int> origin_pb_idx;
  int dump_env = 1;
  const char* dump_env_str = std::getenv("DUMP_PROFILE_PARALLEL_BLOCK");
  if (dump_env_str != nullptr) {
      dump_env = std::atoi(dump_env_str);
  }
  if(dump_env!=0){
    std::cerr << "Unique ParallelBlock Sequences #################################### \n";
    for (size_t i = 0; i < interval_list.size(); ++i) {
        origin_pb_idx = get_pb_seq_origin_idx(interval_list, i);
        std::cerr << "ParallelBlocks sequence: " 
        << "(" << origin_pb_idx.first << ", " << origin_pb_idx.second << ")"
        << " mapped to "
        << " (" << interval_list[i].first << ", " << interval_list[i].second << ")" << std::endl;
    }
  }
  return;
}

static int get_forward_dot_num(const std::pair<int, int> interval){
  if(interval.first == 0) {
    return interval.second - interval.first + 1;
  }else{
    return interval.second - interval.first;
  }
}

void SaveSegmentMapping(std::vector<std::pair<int, int>>& interval_list, const std::string file_path){
  std::vector<std::pair<int, int>> mapping;
  for (int i = 0; i< interval_list.size(); ++i){
    bool mapped = false;
    for (int j = 0; j < i; ++j){
      if(interval_list[i] == interval_list[j]){
        mapping.push_back(std::make_pair(j, get_forward_dot_num(interval_list[j])));
        mapped = true;
        break;
      }
    }
    if(!mapped){
      mapping.push_back(std::make_pair(i, get_forward_dot_num(interval_list[i])));
    }
  }
  std::string filename = file_path + "/segment_mapping";
  std::ofstream outfile(filename);
  if (!outfile) {
      throw std::runtime_error("Failed to open file for writing.");
  }
  for (const auto& p : mapping) {
      outfile << p.first << "," << p.second << "\n";
  }
  outfile.close();


}

std::string DumpFollowingReindex(const StrategyMap& strategy_map,
                                 const HloInstructionSequence& sequence,
                                 const CostGraph& cost_graph) {
  std::ostringstream os;
  os << "Follow Idx and Strategies Number for each Instruction." << std::endl;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    os << instructions[i]->opcode() << ": "
       << " " << i << " -> " << cost_graph.follow_idx[i] << " "
       << strategy_map.at(instructions[i]).get()->leaf_vector.size()
       << ", reindexing: "
       << "[";
    if (cost_graph.follow_idx[i] != -1) {
      std::vector<int> reindexing = cost_graph.reindexing_vector.at(i);
      for (size_t j = 0; j < reindexing.size() - 1; ++j) {
        os << reindexing[j] << " ";
      }
      os << reindexing.back();
    }
    os << " ]" << std::endl;
  }
  return os.str();
}

std::string DumpNotFollowing(const StrategyMap& strategy_map,
                             const HloInstructionSequence& sequence,
                             const CostGraph& cost_graph,
                             const std::vector<int64_t>& s_val,
                             const std::vector<int>& failed_merged) {
  std::ostringstream os;
  os << "Inst With Not Follow Idx and Choosed Strategies." << std::endl;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
//    if (instructions[i]->opcode() == HloOpcode::kDot) {
      int leaf_id = strategy_map.at(instructions[i])->id;
      int leaf_size =
          strategy_map.at(instructions[i]).get()->leaf_vector.size();
      if (cost_graph.follow_idx[leaf_id] == -1 && leaf_size != 1) {
        os << i << " " << instructions[i]->opcode() << ": "
           << " -> choose " << s_val[leaf_id] << " in " << leaf_size;
        if (std::find(failed_merged.begin(), failed_merged.end(), i) !=
            failed_merged.end()) {
          os << " [failed merged]";
        }
        os << "\n";
      }
 //   }
  }
  return os.str();
}


static void makeDotOpAnno(const HloInstruction* dot_inst, DotOpAnno& dot_anno){
  const HloInstruction* lhs = dot_inst->operand(0);
  const HloInstruction* rhs = dot_inst->operand(1);
  const DotDimensionNumbers& dot_dnums = dot_inst->dot_dimension_numbers();
  std::vector<int> lhs_dim_vector(lhs->shape().dimensions().begin(), lhs->shape().dimensions().end());
  std::vector<int> rhs_dim_vector(rhs->shape().dimensions().begin(), rhs->shape().dimensions().end());
  dot_anno.lhs_dims = lhs_dim_vector;
  dot_anno.rhs_dims = rhs_dim_vector;
  dot_anno.lhs_contracting_dim = dot_dnums.lhs_contracting_dimensions()[0];
  dot_anno.rhs_contracting_dim = dot_dnums.rhs_contracting_dimensions()[0];

  return;
}

void ParallelBlockSeq::dumpFingerprint() const {
  // dump all nodes
  for (int i = 0; i < fingerprint.getNodeSize(); ++i){
    std::cerr << "node " << i << " " << fingerprint.getNode(i) << " shape: " 
      << V2S(fingerprint.getNode(i)->annotation.annotation.lhs_dims) << " " 
      <<V2S(fingerprint.getNode(i)->annotation.annotation.rhs_dims) << "\n";
      for (auto e : fingerprint.getNode(i)->outgoingEdgeIndexs) {
        std::cerr << " " << fingerprint.getEdge(e)->src << " -> " << fingerprint.getEdge(e)->dst << "\n";
      }

  }
  // dump all edges
  for (int i = 0; i < fingerprint.getEdgeSize(); ++i){
    std::cerr << "edge " << i << " src_id " << fingerprint.getEdge(i)->src  
      << " dst_id " << fingerprint.getEdge(i)->dst << " dep_dims " << V2S(fingerprint.getEdge(i)->annotation.annotation.dep_dims) << "\n";
  }

}
void ParallelBlockSeq::buildFingerprint(const InstructionDepthMap& depth_map) {
  this->fingerprint = PBSequenceFingerprint();
  std::vector<HloInstruction*> node_list;
  // add all tensor contraction operators to fingerprint
  for (const auto& pb : this->pbs){
    for (const auto inst : pb.first){
      if (inst->opcode() != HloOpcode::kDot || inst->opcode() == HloOpcode::kConvolution){
        continue;
      }
      else{
        DotOpAnno dot_anno;
        makeDotOpAnno(inst, dot_anno);
        FingerprintAnnotation<DotOpAnno> tensor_contraction_anno;
        tensor_contraction_anno.annotation = dot_anno;
        this->fingerprint.addNode(tensor_contraction_anno);
        node_list.push_back(inst);
      }
    }
  }
  // sort node list by instruciton depth.
  std::sort(node_list.begin(), node_list.end(), [&](HloInstruction* a, HloInstruction* b){
    if(depth_map.at(a) == depth_map.at(b)){
      return a->shape().dimensions() < b->shape().dimensions();
    }
    else {
      return depth_map.at(a) < depth_map.at(b);
    }
  }
  );
  // for each node in node_list, find its lhs_input, rhs_input, and output's users that in the list.
  for (int64_t index = 0; index < node_list.size(); index++){
    const auto inst = node_list[index];
    if(!(inst->opcode() == HloOpcode::kDot|| inst->opcode() == HloOpcode::kConvolution)) {
      LOG(FATAL) << "Not Support Reduce Operator.";
    }
    const HloInstruction* lhs = inst->operand(0);
    const HloInstruction* rhs = inst->operand(1);
    std::vector<std::vector<const HloInstruction*>> paths;
    FindOneHopPathsForMultiDst(lhs, node_list, paths);
    FindOneHopPathsForMultiDst(rhs, node_list, paths);
    FindOneHopPathsForMultiDst(inst, node_list, paths);
  
    // build edge for each path.
    for (auto path : paths){
      OpEdgeAnno edge_anno;
      const HloInstruction* src;
      const HloInstruction* dst = path[path.size()-1];
      // get pos
      if (path[0]->opcode()!=HloOpcode::kDot && path[0]->opcode()!=HloOpcode::kConvolution){
        src = path[0]->users()[0];
        if (src->opcode()!=HloOpcode::kDot && src->opcode()!=HloOpcode::kConvolution){
          LOG(FATAL) << "Src not input or output of a tensor contraction.";
        }
        edge_anno.src_pos = src->operand(0)==path[0] ? 0 : 1;
      }
      else {
        src = path[0];
        edge_anno.src_pos = 2;
      }
      edge_anno.dst_pos = dst->operand(0)==path[path.size()-2] ? 0 : 1;

      if (src == dst) {continue;}

      // get src and dst id in list
      auto src_it = std::find(node_list.begin(), node_list.end(), src);
      auto dst_it = std::find(node_list.begin(), node_list.end(), dst);
      int src_id = std::distance(node_list.begin(), src_it);
      int dst_id = std::distance(node_list.begin(), dst_it);
      std::vector<int> dep_dims(path.front()->shape().dimensions_size());
      if (path[path.size()-1]->opcode()==HloOpcode::kDot||path[path.size()-1]->opcode()==HloOpcode::kConvolution){
        path.pop_back();
      }
      std::vector<HloInstruction*> path_;
      for (const auto inst : path){
        path_.push_back(const_cast<HloInstruction*>(inst));
      }
      InferDepsFromPath(path_, dep_dims);
      edge_anno.dep_dims = dep_dims;

      FingerprintAnnotation<OpEdgeAnno> EdgeAnnotation;
      EdgeAnnotation.annotation = edge_anno;
      this->fingerprint.addDepEdge(src_id, dst_id, EdgeAnnotation);
    }
  }
  // dumpFingerprint();
  return;
}

ParallelBlockSeq::ParallelBlockSeq(const HloInstructionSequence& sequence, 
        const ParallelBlockConfig& pb_config, const InstructionDepthMap& depth_map,
        const std::vector<ParallelBlock>& parallel_blocks, const int start_idx, const int seq_length)
        : ins_seq(sequence), pb_config(pb_config), start_idx(start_idx), seq_len(seq_length){
    auto _start = parallel_blocks.begin() + start_idx;
    auto _end = parallel_blocks.end();
    if (start_idx + seq_length < parallel_blocks.size()) {
        _end = parallel_blocks.begin() + start_idx + seq_length;
    }
    pbs = std::vector<ParallelBlock>(_start, _end);
    buildFingerprint(depth_map); 
}


void IntermediateFilePrinter::print(const std::string& file_name, const std::string& data) {
    if (dump_) {
      std::ofstream out_;
      std::string file_path = prefix_ + "/" + file_name + ".txt";
      out_.open(file_path, std::ios::out | std::ios::trunc);
      out_ << data;
      out_.close();
    }
}

  template <typename Fn, typename... Args>
  void IntermediateFilePrinter::print(const std::string& file_name, Fn fn, Args... args) {
    if (dump_) {
      std::ofstream out_;
      std::string file_path = prefix_ + "/" + file_name + ".txt";
      out_.open(file_path, std::ios::out | std::ios::trunc);
      out_ << fn(args...);
      out_.close();
    }
  }

bool InCommFreeRegions(const HloInstruction* inst,
                       const CommunicationFreeRegions& regions) {
  // other constraints.
  for (auto region : regions) {
    if (region.count(inst)) return true;
  }
  return false;
}

}  // namespace spmd
}  // namespace xla
