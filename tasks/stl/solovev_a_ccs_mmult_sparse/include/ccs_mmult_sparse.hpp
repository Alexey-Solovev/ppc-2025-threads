#pragma once

#include <atomic>
#include <algorithm>
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

  MatrixInCcsSparse(int r_nn = 0, int c_nn = 0, int n_zz = 0) {
    c_n = c_nn;
    r_n = r_nn;
    n_z = n_zz;
    row.resize(n_z);
    col_p.resize(r_n + 1);
    val.resize(n_z);
  }
};

class SeqMatMultCcs : public ppc::core::Task {
 public:
  explicit SeqMatMultCcs(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void worker_loop(solovev_a_matrix_stl::SeqMatMultCcs* self);

 private:
  MatrixInCcsSparse *M1_, *M2_, *M3_;

  std::vector<std::thread> workers_;
  std::once_flag init_flag_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<int> next_col_;
  int phase_ = 0;
  int r_n_, c_n_;

  std::vector<int> counts_;
};
}  // namespace solovev_a_matrix_stl