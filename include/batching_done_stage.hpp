/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#ifndef BATCHING_DONE_STAGE_HPP_
#define BATCHING_DONE_STAGE_HPP_

#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <cassert>

#include "infer_task.hpp"
#include "infer_resource.hpp"


struct AutoSetDone {
  explicit AutoSetDone(const std::shared_ptr<std::promise<void>>& p,
                       std::shared_ptr<FrameInfo> data)
      : p_(p), data_(data) {}
  ~AutoSetDone() {
    p_->set_value();
  }
  std::shared_ptr<std::promise<void>> p_;
  std::shared_ptr<FrameInfo> data_;
};  // struct AutoSetDone

using BatchingDoneInput = std::vector<std::pair<std::shared_ptr<FrameInfo>, std::shared_ptr<AutoSetDone>>>;

class BatchingDoneStage {
 public:
  BatchingDoneStage() = default;
  BatchingDoneStage(uint32_t batchsize)
      : batchsize_(batchsize) {}
  virtual ~BatchingDoneStage() {}
  virtual std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) = 0;
 protected:
  uint32_t batchsize_ = 0;
};  // class BatchingDoneStage


class H2DBatchingDoneStage : public BatchingDoneStage {
 public:
  H2DBatchingDoneStage(uint32_t batchsize,
                       std::shared_ptr<IOResource> cpu_input_res, 
                       std::shared_ptr<IOResource> mlu_input_res)
      : BatchingDoneStage(batchsize), cpu_input_res_(cpu_input_res), mlu_input_res_(mlu_input_res) {}
  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) override {
    std::vector<InferTaskSptr> tasks;
    InferTaskSptr task;

    QueuingTicket cpu_input_res_ticket = cpu_input_res_->PickUpNewTicket();
    QueuingTicket mlu_input_res_ticket = mlu_input_res_->PickUpNewTicket();

    task = std::make_shared<InferTask>([cpu_input_res_ticket, mlu_input_res_ticket, this, finfos]() -> int {
      QueuingTicket cir_ticket = cpu_input_res_ticket;
      QueuingTicket mir_ticket = mlu_input_res_ticket;

      // waiting for schedule
      IOResValue cpu_value = this->cpu_input_res_->WaitResourceByTicket(&cir_ticket);
      IOResValue mlu_value = this->mlu_input_res_->WaitResourceByTicket(&mir_ticket);

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      assert(finfos.size() == batchsize_);

      for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
        std::cout << "H2DBatchingDoneStage BatchingDone: " << finfos[bidx].first->time_stamp_ << std::endl;
      }
      this->cpu_input_res_->DeallingDone();
      this->mlu_input_res_->DeallingDone();
      return 0;
    });
    tasks.push_back(task);
    return tasks;
 } 
 private:
  std::shared_ptr<IOResource> cpu_input_res_;
  std::shared_ptr<IOResource> mlu_input_res_;
};  // class H2DBatchingDoneStage


class InferBatchingDoneStage : public BatchingDoneStage {
 public:
  InferBatchingDoneStage(uint32_t batchsize,
                         std::shared_ptr<IOResource> mlu_input_res,
                         std::shared_ptr<IOResource> mlu_output_res):
      BatchingDoneStage(batchsize), mlu_input_res_(mlu_input_res), mlu_output_res_(mlu_output_res) {}
  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) override {
    std::vector<InferTaskSptr> tasks;
    InferTaskSptr task;

    QueuingTicket mlu_input_res_ticket = mlu_input_res_->PickUpNewTicket();
    QueuingTicket mlu_output_res_ticket = mlu_output_res_->PickUpNewTicket();
    task = std::make_shared<InferTask>([mlu_input_res_ticket, mlu_output_res_ticket, this, finfos]() -> int {
      QueuingTicket mir_ticket = mlu_input_res_ticket;
      QueuingTicket mor_ticket = mlu_output_res_ticket;
      IOResValue mlu_input_value = this->mlu_input_res_->WaitResourceByTicket(&mir_ticket);
      IOResValue mlu_output_value = this->mlu_output_res_->WaitResourceByTicket(&mor_ticket);

      std::this_thread::sleep_for(std::chrono::milliseconds(800));
      assert(finfos.size() == batchsize_);
      for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
        std::cout << "InferBatchingDoneStage BatchingDone: " << finfos[bidx].first->time_stamp_ << std::endl;
      }
      this->mlu_input_res_->DeallingDone();
      this->mlu_output_res_->DeallingDone();
      return 0;
    });
    tasks.push_back(task);
    return tasks;
  }
 private:
  std::shared_ptr<IOResource> mlu_input_res_;
  std::shared_ptr<IOResource> mlu_output_res_;
};  // class InferBatchingDoneStage

class D2HBatchingDoneStage : public BatchingDoneStage {
 public:
  D2HBatchingDoneStage(uint32_t batchsize,
                       std::shared_ptr<IOResource> mlu_output_res,
                       std::shared_ptr<IOResource> cpu_output_res)
      : BatchingDoneStage(batchsize), mlu_output_res_(mlu_output_res), cpu_output_res_(cpu_output_res) {}

  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) override {
    std::vector<InferTaskSptr> tasks;
    InferTaskSptr task;
    QueuingTicket mlu_output_res_ticket = mlu_output_res_->PickUpNewTicket();
    QueuingTicket cpu_output_res_ticket = cpu_output_res_->PickUpNewTicket();
    task = std::make_shared<InferTask>([mlu_output_res_ticket, cpu_output_res_ticket, this, finfos]() -> int {
      QueuingTicket mor_ticket = mlu_output_res_ticket;
      QueuingTicket cor_ticket = cpu_output_res_ticket;
      IOResValue mlu_output_value = this->mlu_output_res_->WaitResourceByTicket(&mor_ticket);
      IOResValue cpu_output_value = this->cpu_output_res_->WaitResourceByTicket(&cor_ticket);

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      assert(finfos.size() == batchsize_);
      for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
        std::cout << "D2HBatchingDoneStage BatchingDone: " << finfos[bidx].first->time_stamp_ << std::endl;
      }

      this->mlu_output_res_->DeallingDone();
      this->cpu_output_res_->DeallingDone();
      return 0;
    });
    tasks.push_back(task);
    return tasks;
  }

 private:
  std::shared_ptr<IOResource> mlu_output_res_;
  std::shared_ptr<IOResource> cpu_output_res_;
};  // class D2HBatchingDoneStage

class PostprocessingBatchingDoneStage : public BatchingDoneStage {
 public:
  PostprocessingBatchingDoneStage(uint32_t batchsize,
                                  std::shared_ptr<IOResource> cpu_output_res)
      : BatchingDoneStage(batchsize), cpu_output_res_(cpu_output_res) {}

  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) override {
    std::vector<InferTaskSptr> tasks;
    assert(finfos.size() == batchsize_);
    for (int bidx = 0; bidx < static_cast<int>(finfos.size()); ++bidx) {
      auto finfo = finfos[bidx];
      QueuingTicket cpu_output_res_ticket;
      if (0 == bidx) {
        cpu_output_res_ticket = this->cpu_output_res_->PickUpNewTicket(true);
      } else {
        cpu_output_res_ticket = this->cpu_output_res_->PickUpTicket(true);
      }
      InferTaskSptr task =
        std::make_shared<InferTask>([cpu_output_res_ticket, this, finfo, bidx]() -> int {
          QueuingTicket cor_ticket = cpu_output_res_ticket;
          IOResValue cpu_output_value = this->cpu_output_res_->WaitResourceByTicket(&cor_ticket);
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          std::cout << "PostprocessingBatchingDoneStage, bidx: " << bidx << ", time_stamp: " << finfo.first->time_stamp_ << std::endl;
          this->cpu_output_res_->DeallingDone();
          return 0;
        });
      tasks.push_back(task);
    }  // end for (bidx)
    return tasks;
 }
 private:
  std::shared_ptr<IOResource> cpu_output_res_ = nullptr;
};  // class PostprocessingBatchingDoneStage


#endif  // MODULES_INFERENCE_SRC_BATCHING_DONE_STAGE_HPP_
