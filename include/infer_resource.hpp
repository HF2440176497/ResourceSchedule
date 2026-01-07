#ifndef INFER_RESOURCE_HPP_
#define INFER_RESOURCE_HPP_

#include <memory>
#include <vector>

#include "queuing_server.hpp"


template <typename RetT>
class InferResource : public QueuingServer {
 public:
  InferResource(uint32_t batchsize) : batchsize_(batchsize) {}
  virtual ~InferResource() {}
  virtual void Init() {}
  virtual void Destroy() {}
  RetT WaitResourceByTicket(QueuingTicket* pticket) {
    WaitByTicket(pticket);
    return value_;
  }
  RetT GetDataDirectly() const { return value_; }

 protected:
  const uint32_t batchsize_ = 0;
  RetT value_;  // 在外界初始化时进行赋值
};  // class InferResource

// 模拟分配的内存资源
struct IOResValue {
  struct OneData
  {
    std::vector<int> datas;  // size == batch_size
    uint32_t batchsize = 0;
  };
  std::vector<OneData> datas;  // size == 1
};  // struct IOResValue


class IOResource : public InferResource<IOResValue> {
 public:
  IOResource(uint32_t batchsize) : InferResource<IOResValue>(batchsize) {}
  void Init() override {
    value_ = Allocate(batchsize_);
  }
  IOResValue Allocate(uint32_t batchsize) {
    IOResValue res;
    res.datas.reserve(1);
    res.datas[0].datas = std::vector<int>(batchsize, 0);
    res.datas[0].batchsize = batchsize;
    return res;
  }
};  // class IOResource

#endif  // INFER_RESOURCE_HPP_