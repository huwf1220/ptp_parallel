#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_OPT_EXP_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_OPT_EXP_H_

#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"
#include "tensorflow/compiler/xla/service/spmd/reindexing.pb.h"


namespace xla {
namespace spmd {

#define TF_LOG_VECTOR(INFO, V, NAME) \
  LOG(INFO) << NAME << ": " << absl::StrJoin(V, " ") << "\n"

class IntermediateFilePrinter {
public:
  IntermediateFilePrinter(const std::string& prefix, bool dump)
      : dump_(dump), prefix_(prefix) {}

void print(const std::string& file_name, const std::string& data); 

template <typename Fn, typename... Args>
void print(const std::string& file_name, Fn fn, Args... args);

private:
  bool dump_;
  std::string prefix_;
};

template <typename T>
struct FingerprintAnnotation {
    T annotation;  
    bool equal(const FingerprintAnnotation& other) const {
        return annotation == other.annotation;  
    }
    bool operator==(const FingerprintAnnotation& other) const {
        return equal(other);
    }
    bool operator<(const FingerprintAnnotation& other) const {
        return annotation < other.annotation;
    }
};

template <typename T>
std::string V2S_(const std::vector<T>& vec) {
  std::stringstream ss;
  for (const auto& elem : vec) {
    ss << elem << ", ";
  }
  std::string result = ss.str();
  return result;
}

template <typename T1, typename T2>
class DepDAGFingerprint{
public:
    struct DepEdge;
    struct ContractOp {
        FingerprintAnnotation<T1> annotation;
        std::vector<int> outgoingEdgeIndexs;
        ContractOp(const FingerprintAnnotation<T1>& annotation) : annotation(annotation) {}
    };
    struct DepEdge {
        int src;
        int dst;
        FingerprintAnnotation<T2> annotation;

        DepEdge(int src, int dst, const FingerprintAnnotation<T2>& annotation)
            : src(src), dst(dst), annotation(annotation) {}
    };

    ContractOp* addNode(const FingerprintAnnotation<T1>& annotation) {
        nodes.emplace_back(annotation);
        return &nodes.back(); 
    }

    const ContractOp* getNode(const int index) const {return &nodes[index];}
    const DepEdge* getEdge(const int index) const {return &edges[index];}

    int64_t getNodeSize() const {return nodes.size();}
    int64_t getEdgeSize() const {return edges.size();}

    void addDepEdge(int src, int dst, const FingerprintAnnotation<T2>& annotation){
        edges.emplace_back(src, dst, annotation);
        nodes[src].outgoingEdgeIndexs.push_back(getEdgeSize()-1);
    }

    const std::vector<ContractOp>* getNodeList() const {
        return &nodes;
    } 
    const std::vector<DepEdge>* getEdgeList() const {
        return &edges;
    } 
    bool match(const DepDAGFingerprint<T1, T2>& other_fingerprint){
        const std::vector<ContractOp>* a_nodes = this->getNodeList();
        const std::vector<ContractOp>* b_nodes = other_fingerprint.getNodeList();
        const std::vector<DepEdge>* a_edges = this->getEdgeList();
        const std::vector<DepEdge>* b_edges = other_fingerprint.getEdgeList();
        if (a_nodes->size() != b_nodes->size() || a_edges->size() != b_edges->size()){
            return false;
        }
        // for each node, check annotation and edges
        for (int idx = 0; idx < a_nodes->size(); ++idx){ 
            if (!((*a_nodes)[idx].annotation.annotation == (*b_nodes)[idx].annotation.annotation)){
                //std::cerr << "Node " << idx << " Annotation Did Not Matched\n";
                return false;
            }
            std::vector<int> a_node_edge_indexs((*a_nodes)[idx].outgoingEdgeIndexs);
            std::vector<int> b_node_edge_indexs((*b_nodes)[idx].outgoingEdgeIndexs);
            if (a_node_edge_indexs.size()!=b_node_edge_indexs.size()){
                //std::cerr << "Node " << idx << " Outgoing edges size Did Not Matched\n";
                return false;
            }
    
            for (int edge_idx = 0; edge_idx < a_node_edge_indexs.size(); ++edge_idx){
                const DepEdge* self_edge = getEdge(a_node_edge_indexs[edge_idx]);
                const DepEdge* other_edge = other_fingerprint.getEdge(b_node_edge_indexs[edge_idx]);
                if(!(self_edge->dst== other_edge->dst)){
                    //std::cerr << "Node " << idx << " Edge " << edge_idx <<" Dst annotaition Did Not Matched\n";
                    return false;
                }
                if (!(self_edge->annotation == other_edge->annotation)){
                    //std::cerr << "Node " << idx << " Edge " << edge_idx <<" edge annotaition Did Not Matched\n";
                    return false;
                }
            }
        }
        return true;
    }
  
private:
    std::vector<ContractOp> nodes;  
    std::vector<DepEdge> edges;  
};

struct DotOpAnno {
 std::vector<int> lhs_dims;
 std::vector<int> rhs_dims;
 int lhs_contracting_dim;
 int rhs_contracting_dim;
 bool operator==(const DotOpAnno& other) const {
    return lhs_dims == other.lhs_dims && rhs_dims == other.rhs_dims 
        && lhs_contracting_dim == other.lhs_contracting_dim
        && rhs_contracting_dim == other.rhs_contracting_dim;
 }

bool operator<(const DotOpAnno& other) const {
    if (lhs_dims < other.lhs_dims) return true;
    if (lhs_dims > other.lhs_dims) return false;
    if (rhs_dims < other.rhs_dims) return true;
    if (rhs_dims > other.rhs_dims) return false;
    if (lhs_contracting_dim < other.lhs_contracting_dim) return true;
    if (lhs_contracting_dim > other.lhs_contracting_dim) return false;
    return rhs_contracting_dim < other.rhs_contracting_dim;
}
};

struct OpEdgeAnno {
std::vector<int> dep_dims;
int src_pos;//0-lhs input, 1-rhs input, 2-output
int dst_pos;//0-lhs input, 1-rhs input
 bool operator==(const OpEdgeAnno& other) const {
    return dep_dims == other.dep_dims
        && src_pos == other.src_pos
        && dst_pos == other.dst_pos;
 }
};

using PBSequenceFingerprint = DepDAGFingerprint<DotOpAnno, OpEdgeAnno>;

class ParallelBlockSeq {
public:
    ParallelBlockSeq(const HloInstructionSequence& sequence, 
        const ParallelBlockConfig& pb_config, 
        const InstructionDepthMap& depth_map,
        const std::vector<ParallelBlock>& parallel_blocks, const int start_idx, const int seq_length);
    void buildFingerprint(const InstructionDepthMap& depth_map);
    void dumpFingerprint() const;
    bool match(const ParallelBlockSeq& seq_) {
        return this->fingerprint.match(seq_.fingerprint);
    }
    int getSeqLength() {return seq_len;}
    int getStartIdx() {return start_idx;}

private:
    const HloInstructionSequence& ins_seq;
    const ParallelBlockConfig& pb_config; 
    std::vector<ParallelBlock> pbs;
    PBSequenceFingerprint fingerprint;
    int seq_len;
    int start_idx;
};


// Build parallel blocks for the given HloInstructionSequence.
void BuildParallelBlocks(const HloInstructionSequence& sequence,
                         const LeafStrategies& leaf_strategies,
                         const StrategyMap& strategy_map,
                         const ClusterEnvironment& cluster_env,
                         const ParallelBlockConfig& pb_config,
                         std::vector<ParallelBlock>& parallel_blocks,
                         ReindexItemList& reindex_arr,
                         AutoShardingSolverOption& solver_option, std::vector<int>& set_to_one);

void LoadParallelBlocks(std::string loaded_pb,
                        const HloInstructionSequence& sequence,
                        std::vector<ParallelBlock>& parallel_blocks);

void ResetStrategiesForParallelBlocks(
    const HloInstructionSequence& sequence,
    const std::vector<ParallelBlock>& parallel_blocks,
    LeafStrategies& leaf_strategies, StrategyMap& strategy_map, std::vector<int>& set_to_one, AutoShardingSolverOption& solver_option);

void ReduceEdgesForParallelBlocks(
    const HloInstructionSequence& sequence,
    const LeafStrategies& leaf_strategies, const StrategyMap& strategy_map,
    CostGraph& cost_graph, const std::vector<ParallelBlock>& parallel_blocks, std::vector<int>& failed_merged);

void InferSolutionForRemainInsts(const HloInstructionSequence& sequence, 
    const LeafStrategies& leaf_strategies, const StrategyMap& strategy_map, const CostGraph& cost_graph,
    const std::vector<ParallelBlock>& parallel_blocks, std::vector<long int>& solution_vector, const std::string reindex_file);

void CheckEdgeCosts(const LeafStrategies& leaf_strategies, const CostGraph& cost_graph, std::vector<long int>& s_val);

void SaveReindexToFile(ReindexItemList& reindex_list, const std::string& file_name, bool append);

void LoadReindexFromFile(ReindexItemList& reindex_list, const std::string& file_path);

void SaveReindexFile(const HloInstructionSequence& sequence, const CostGraph& cost_graph, 
    const LeafStrategies& leaf_strategies, const StrategyMap& strategy_map, 
    const std::string file_path, const std::vector<int>& failed_merge, 
    const std::vector<ParallelBlock>& parallel_blocks, ReindexItemList& reindex_list);

void ExtractPBSequences(const HloInstructionSequence& sequence, const ParallelBlockConfig& pb_config, 
        const InstructionDepthMap& depth_map, AutoShardingSolverOption& solver_option, const std::vector<ParallelBlock>& parallel_blocks, 
        std::vector<int> pbs_idx, std::vector<std::pair<int, int>>& interval_list);

void SaveSegmentMapping(std::vector<std::pair<int, int>>& interval_list, const std::string file_path);

void LoadDotPairs(std::string str, DotPairs& pairs);

std::string DumpFollowingReindex(const StrategyMap& strategy_map,
                                 const HloInstructionSequence& sequence,
                                 const CostGraph& cost_graph);
std::string DumpNotFollowing(const StrategyMap& strategy_map,
                             const HloInstructionSequence& sequence,
                             const CostGraph& cost_graph,
                             const std::vector<int64_t>& s_val, const std::vector<int>& failed_merged);

std::string DumpSVFile(const CostGraph& cost_graph, const StrategyMap& strategy_map);
//


/* ====================================================================
 * experimental code for auto-sharding optimization.
   ==================================================================== */
// Clustered dot/conv pairs after inter-operator analysis.
using CommunicationFreeOpPairs = std::vector<std::pair<int, int>>;
// Record a set of instrcutions that lie in a communication free path
using CommunicationFreeRegions =
    std::vector<absl::flat_hash_set<HloInstruction*>>;
// Record all inverse following list for key operator to its input parameters.
using FollowingLists =
    std::vector<std::pair<HloInstruction*, std::vector<HloInstruction*>>>;

[[deprecated("CommunicationFreeRegion related code is deprecated. Use the parallel_block instead.")]]
bool InCommFreeRegions(const HloInstruction* inst,
                       const CommunicationFreeRegions& regions);


}  // namespace spmd
}  // namespace xla


#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_OPT_EXP_H_
