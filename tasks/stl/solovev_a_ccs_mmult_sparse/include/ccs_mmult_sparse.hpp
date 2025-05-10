#pragma once

#include <algorithm>
#include <atomic>
#include <complex>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovev_a_matrix_stl {

struct MatrixInCcsSparse {
  std::vector<std::complex<double>> val;
  std::vector<int> row;
  std::vector<int> col_p;

  int r_n;
  int c_n;
  int n_z;

  MatrixInCcsSparse(int r_nn = 0, int c_nn = 0, int n_zz = 0)
      : r_n(r_nn), c_n(c_nn), n_z(n_zz), val(n_zz), row(n_zz), col_p(c_nn + 1) {}
};

class SeqMatMultCcs : public ppc::core::Task {
 public:
  explicit SeqMatMultCcs(std::shared_ptr<ppc::core::TaskData> task_data)
      : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void worker_loop(SeqMatMultCcs* self);

 private:
  MatrixInCcsSparse *M1_, *M2_, *M3_;

  std::vector<std::thread> workers_;
  std::once_flag init_flag_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<int> next_col_{0};
  std::atomic<int> completed_{0};
  int phase_ = 0;
  int r_n_ = 0, c_n_ = 0;
  bool terminate_ = false;

  std::vector<int> counts_;
};

}  // namespace solovev_a_matrix_stl
