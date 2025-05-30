#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/ermolaev_v_graham_scan/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
ppc::core::TaskDataPtr CreateTaskData(std::vector<ermolaev_v_graham_scan_all::Point>& input,
                                      std::vector<ermolaev_v_graham_scan_all::Point>& output) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }
  return task_data;
}

std::vector<ermolaev_v_graham_scan_all::Point> CreateInput(int count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(-100, 100);

  std::vector<ermolaev_v_graham_scan_all::Point> input;
  input.reserve(count);
  for (int i = 0; i < count; i++) {
    input.emplace_back(dis(gen), dis(gen));
  }

  return input;
}

bool ValidateConvexHull(const std::vector<ermolaev_v_graham_scan_all::Point>& hull, const size_t size) {
  if (hull.size() < 3) {
    return false;
  }

  for (size_t i = 0; i < size; ++i) {
    const auto& p1 = hull[i];
    const auto& p2 = hull[(i + 1) % size];
    const auto& p3 = hull[(i + 2) % size];

    int cross = ((p2.x - p1.x) * (p3.y - p1.y)) - ((p3.x - p1.x) * (p2.y - p1.y));
    if (cross < 0) {
      return false;
    }
  }

  return true;
}
}  // namespace

TEST(ermolaev_v_graham_scan_all, run_pipeline) {
  boost::mpi::communicator world;
  constexpr int kCount = 2500000;

  auto input = CreateInput(kCount);
  std::vector<ermolaev_v_graham_scan_all::Point> output(kCount);

  auto task_data_all = CreateTaskData(input, output);
  auto test_task_alluential = std::make_shared<ermolaev_v_graham_scan_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_alluential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(ValidateConvexHull(output, task_data_all->outputs_count[0]));
  }
}

TEST(ermolaev_v_graham_scan_all, run_task) {
  boost::mpi::communicator world;
  constexpr int kCount = 2500000;

  auto input = CreateInput(kCount);
  std::vector<ermolaev_v_graham_scan_all::Point> output(kCount);

  auto task_data_all = CreateTaskData(input, output);
  auto test_task_alluential = std::make_shared<ermolaev_v_graham_scan_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_alluential);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(ValidateConvexHull(output, task_data_all->outputs_count[0]));
  }
}
