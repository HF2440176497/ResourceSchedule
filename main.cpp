
#include <vector>
#include <mutex>
#include <memory>
#include <future>

#include "infer_resource.hpp"
#include "batching_stage.hpp"
#include "batching_done_stage.hpp"
#include "infer_thread_pool.hpp"
#include "infer_trans_data_helper.hpp"


uint32_t batchsize = 4;

auto cpu_input_res_ = std::make_shared<IOResource>(batchsize);
auto cpu_output_res_ = std::make_shared<IOResource>(batchsize);
auto mlu_input_res_ = std::make_shared<IOResource>(batchsize);
auto mlu_output_res_ = std::make_shared<IOResource>(batchsize);

auto batching_stage_ = std::make_shared<IOBatchingStage>(batchsize, cpu_input_res_);

std::vector<std::shared_ptr<BatchingDoneStage>> batching_done_stages_ =
{ std::make_shared<H2DBatchingDoneStage>(batchsize, cpu_input_res_, mlu_input_res_),
 std::make_shared<InferBatchingDoneStage>(batchsize, mlu_input_res_, mlu_output_res_),
 std::make_shared<D2HBatchingDoneStage>(batchsize, mlu_output_res_, cpu_output_res_),
 std::make_shared<PostprocessingBatchingDoneStage>(batchsize, cpu_output_res_) };

BatchingDoneInput batched_finfos_{ };
auto tp_ = std::make_shared<InferThreadPool>();
auto trans_helper_ = std::make_shared<InferTransDataHelper>(batchsize);

int64_t current_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

/**
 * @brief 攒够一个批次的数据，进行处理
 */
void BatchingDone() {
  batching_stage_->Reset();
  if (!batched_finfos_.empty()) {
    for (auto& it : batching_done_stages_) {
      auto tasks = it->BatchingDone(batched_finfos_);
      tp_->SubmitTask(tasks);
    }
    batched_finfos_.clear();
  }
}

ResultWaitingCard FeedData(std::shared_ptr<FrameInfo> finfo) {
    auto ret_promise = std::make_shared<std::promise<void>>();
    ResultWaitingCard card(ret_promise);
    auto auto_set_done = std::make_shared<AutoSetDone>(ret_promise, finfo);

    InferTaskSptr task = batching_stage_->Batching(finfo);
    tp_->SubmitTask(task);

    batched_finfos_.push_back(std::make_pair(finfo, auto_set_done));
    if (batched_finfos_.size() == batchsize) {
        BatchingDone();
    }
    return card;
}

int main() {
  tp_->Init(batchsize * 3 + 4);

  cpu_input_res_->Init();
  cpu_output_res_->Init();
  mlu_input_res_->Init();
  mlu_output_res_->Init();

  int num_batch = 4;
  for (int i = 0; i < num_batch; i++) {
    for (int j = 0; j < batchsize; ++j) {
      std::shared_ptr<FrameInfo> finfo = std::make_shared<FrameInfo>();
      finfo->batch_index = i;
      finfo->item_index = j;
      ResultWaitingCard card = FeedData(finfo);

      // TODO: 人为制造的延迟 便于打印
      if (i == 0 && j <= 1) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      trans_helper_->SubmitData(std::make_pair(finfo, card));
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(10));
  tp_->Destroy();
  trans_helper_.reset();
}