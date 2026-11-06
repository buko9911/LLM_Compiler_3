#include "npu/Decode/PimMetadata.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <limits>
#include <set>

namespace npu::decode {
using namespace ::npu::pim;
llvm::json::Value toJSON(const Program &program) {

  llvm::json::Array a, p, ph;
  for (const Activation &x : program.activations)
    a.push_back(llvm::json::Object{{"k", x.k}, {"base", x.base}, {"parts", llvm::json::Array(x.parts)}});
  for (const Projection &x : program.projections) {
    llvm::json::Object o{{"weight", x.name}, {"n", x.n}, {"k", x.k}, {"n_pad", x.nPad},
                         {"stream_weight_base", llvm::json::Array(x.streamWeightBase)}, {"output_base", x.outputBase},
                         {"acts", llvm::json::Array(x.activations)}, {"stream_part", llvm::json::Array(x.streamPart)},
                         {"compute_first", x.computeFirst}, {"compute_count", x.computeCount}};
    // Stream shape, written only when it is not the 512-row stream over all of K.
    if (x.accs != kAccs) o["accs"] = x.accs;
    if (x.splitK != 1) o["split_k"] = x.splitK;
    if (x.tailBlocks) { o["tail_accs"] = x.tailAccs; o["tail_blocks"] = x.tailBlocks; }
    if (x.inputsPerSweep != kInTiles) o["inputs_per_sweep"] = x.inputsPerSweep;
    if (x.common) {
      o["common"] = true;
      o["stream_order"] = llvm::json::Array(x.streamOrder);
      llvm::json::Array segments;
      for (const auto &seg : x.segments) segments.push_back(llvm::json::Array{seg.first, seg.second});
      o["segments"] = std::move(segments);
    }
    p.push_back(std::move(o));
  }
  for (auto [f, c] : program.phases) ph.push_back(llvm::json::Array{f, c});
  llvm::json::Object common;
  if (program.commonLayout) common["common_layout"] = program.commonLayout;
  if (program.commonTurnSweeps) common["common_turn_sweeps"] = program.commonTurnSweeps;
  llvm::json::Object out{{"channels", program.channels}, {"partitions", program.partitions}, {"kernel_row", program.kernelRow},
                            {"weight_layout", program.weightLayout()}, {"bankset_pingpong_lead", program.banksetPingPongLead},
                            {"next_w", llvm::json::Array(program.nextWeight)},
                            {"next_a", program.nextActivation}, {"next_o", program.nextOutput},
                            {"peak_a", program.activationTiles()}, {"peak_o", program.outputTiles()},
                            {"reuse_buffers", program.reuseBuffers}, {"acts", std::move(a)},
                            {"projections", std::move(p)}, {"phases", std::move(ph)}};
  for (auto &kv : common) out[kv.first] = std::move(kv.second);
  return out;
}

Program fromJSON(const llvm::json::Value &v) {
  Program pr;
  const llvm::json::Object *o = v.getAsObject();
  pr.channels = int(o->getInteger("channels").value_or(2));
  pr.partitions = int(*o->getInteger("partitions"));
  pr.kernelRow = *o->getInteger("kernel_row");
  pr.banksetPingPongLead = int(o->getInteger("bankset_pingpong_lead").value_or(0));
  pr.commonLayout = int(o->getInteger("common_layout").value_or(0));
  pr.commonTurnSweeps = int(o->getInteger("common_turn_sweeps").value_or(0));
  if (pr.banksetPingPongLead < 0 || pr.banksetPingPongLead > 63 || pr.banksetPingPongLead == 1 ||
      o->getString("weight_layout").value_or("even-v1") != pr.weightLayout())
    llvm::report_fatal_error("inconsistent PIM weight layout / ping-pong lead");
  pr.nextWeight.clear();
  for (const llvm::json::Value &c : *o->getArray("next_w")) pr.nextWeight.push_back(*c.getAsInteger());
  pr.nextActivation = *o->getInteger("next_a");
  pr.nextOutput = *o->getInteger("next_o");
  if (auto v = o->getInteger("peak_a")) pr.activationPeak = *v;
  if (auto v = o->getInteger("peak_o")) pr.outputPeak = *v;
  if (auto v = o->getBoolean("reuse_buffers")) pr.reuseBuffers = *v;
  for (const llvm::json::Value &x : *o->getArray("acts")) {
    const llvm::json::Object *xo = x.getAsObject();
    Activation a;
    a.k = *xo->getInteger("k");
    a.base = *xo->getInteger("base");
    for (const llvm::json::Value &c : *xo->getArray("parts")) a.parts.push_back(int(*c.getAsInteger()));
    pr.activations.push_back(a);
  }
  for (const llvm::json::Value &x : *o->getArray("projections")) {
    const llvm::json::Object *xo = x.getAsObject();
    Projection p;
    p.name = xo->getString("weight")->str();
    p.n = *xo->getInteger("n");
    p.k = *xo->getInteger("k");
    p.nPad = *xo->getInteger("n_pad");
    if (auto v = xo->getInteger("accs")) p.accs = int(*v);
    if (auto v = xo->getInteger("split_k")) p.splitK = int(*v);
    if (auto v = xo->getInteger("tail_accs")) p.tailAccs = int(*v);
    if (auto v = xo->getInteger("tail_blocks")) p.tailBlocks = *v;
    if (auto v = xo->getInteger("inputs_per_sweep")) p.inputsPerSweep = int(*v);
    if (auto v = xo->getBoolean("common")) p.common = *v;
    if (const llvm::json::Array *a = xo->getArray("stream_order"))
      for (const llvm::json::Value &c : *a) p.streamOrder.push_back(*c.getAsInteger());
    if (const llvm::json::Array *a = xo->getArray("segments"))
      for (const llvm::json::Value &c : *a) {
        const llvm::json::Array *pair = c.getAsArray();
        p.segments.push_back({(*pair)[0].getAsString()->str(), *(*pair)[1].getAsInteger()});
      }
    for (const llvm::json::Value &c : *xo->getArray("stream_weight_base")) p.streamWeightBase.push_back(*c.getAsInteger());
    p.outputBase = *xo->getInteger("output_base");
    for (const llvm::json::Value &c : *xo->getArray("acts")) p.activations.push_back(*c.getAsInteger());
    for (const llvm::json::Value &c : *xo->getArray("stream_part")) p.streamPart.push_back(int(*c.getAsInteger()));
    p.computeFirst = *xo->getInteger("compute_first");
    p.computeCount = *xo->getInteger("compute_count");
    if (!p.streamPart.empty()) p.assignOutputSlots();
    pr.projections.push_back(p);
  }
  for (const llvm::json::Value &x : *o->getArray("phases")) {
    const llvm::json::Array *pa = x.getAsArray();
    pr.phases.push_back({*(*pa)[0].getAsInteger(), *(*pa)[1].getAsInteger()});
  }
  return pr;
}

} // namespace npu::decode
