
#include <vector>
#include <mutex>
#include <memory>
#include <future>

#include "batching_stage.hpp"
#include "batching_done_stage.hpp"
#include "infer_thread_pool.hpp"


class ResultWaitingCard {
 public:
  explicit ResultWaitingCard(std::shared_ptr<std::promise<void>> ret_promise) : promise_(ret_promise) {}
  void WaitForCall() {  // wait for set_value
    promise_->get_future().share().get();
  }
 private:
  std::shared_ptr<std::promise<void>> promise_;
};  // class ResultWaitingCard

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

  int num_frames = 20;  // batch num == 5
  for (int i = 0; i < num_frames; i++) {
    std::shared_ptr<FrameInfo> finfo = std::make_shared<FrameInfo>();
    finfo->time_stamp_ = current_ms();
    FeedData(finfo);
  }

  tp_->Destroy();
}