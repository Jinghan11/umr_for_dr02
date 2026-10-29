// Optional batch wrapper around the installed MuJoCo C API. No approximated geometry.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
static_assert(sizeof(mjtNum) == sizeof(double), "Native geometry requires float64 MuJoCo");
thread_local std::string last_error;
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };
void callbacks_supported(uintptr_t allowed_timer) {
  if (mjcb_passive || mjcb_control || mjcb_contactfilter || mjcb_sensor ||
      reinterpret_cast<uintptr_t>(mjcb_time) != allowed_timer ||
      mjcb_act_dyn || mjcb_act_gain || mjcb_act_bias || mju_user_error || mju_user_warning ||
      mju_user_malloc || mju_user_free)
    throw Unsupported("MuJoCo callbacks/custom handlers require the Python geometry backend");
}
struct Worker {
  mjModel* model = nullptr;
  mjData* data = nullptr;
  std::vector<double> saved_margin, jp1, jr1, jp2, jr2;
  explicit Worker(const mjModel* source) {
    model = mj_copyModel(nullptr, source);
    if (!model) throw std::bad_alloc();
    data = mj_makeData(model);
    if (!data) { mj_deleteModel(model); model = nullptr; throw std::bad_alloc(); }
    try {
      saved_margin.assign(model->geom_margin, model->geom_margin + model->ngeom);
      for (auto* p : {&jp1, &jr1, &jp2, &jr2}) p->resize(3 * model->nv);
    } catch (...) {
      mj_deleteData(data); mj_deleteModel(model); data = nullptr; model = nullptr; throw;
    }
  }
  ~Worker() { if (data) mj_deleteData(data); if (model) mj_deleteModel(model); }
  void forward(const double* pose) {
    std::copy(pose, pose + model->nq, data->qpos);
    const int old_contacts = data->warning[mjWARN_CONTACTFULL].number;
    mj_forward(model, data);
    if (data->warning[mjWARN_CONTACTFULL].number != old_contacts)
      throw std::runtime_error("MuJoCo contact arena overflow; no truncated geometry is returned");
  }
};
struct Row { int a, b; uint8_t kind; double distance, margin; std::array<double, 6> witness; };
struct Result {
  int64_t frames = 0, rows = 0, corners = 0, nv = 0;
  std::vector<int64_t> offsets, corner_rows;
  std::vector<int32_t> pairs;
  std::vector<uint8_t> kinds;
  std::vector<double> distances, margins, witnesses, jacobians, corner_A, corner_b;
};
struct Context {
  int nq, nv, ngeom;
  bool self_enabled;
  uintptr_t allowed_timer;
  double self_threshold, ground_threshold, self_margin, ground_margin, safety;
  std::vector<uint8_t> self_mask, ground_robot, floor_mask, floor_label;
  std::vector<std::unique_ptr<Worker>> workers;
  std::vector<std::thread> threads;
  std::mutex mutex;
  std::condition_variable ready, finished;
  bool stopping = false;
  uint64_t generation = 0;
  size_t remaining = 0, frame_count = 0;
  std::atomic<size_t> next_frame{0};
  std::function<void(size_t, Worker&)> job;
  std::exception_ptr failure;
  Context(const mjModel* m, int count, bool enabled, const uint8_t* sm, const uint8_t* gr,
          const uint8_t* fm, const uint8_t* fl, double st, double gt, double s_margin,
          double g_margin, double safe, uintptr_t timer)
      : nq(m->nq), nv(m->nv), ngeom(m->ngeom), self_enabled(enabled), allowed_timer(timer), self_threshold(st),
        ground_threshold(gt), self_margin(s_margin), ground_margin(g_margin), safety(safe),
        self_mask(sm, sm + ngeom), ground_robot(gr, gr + ngeom), floor_mask(fm, fm + ngeom),
        floor_label(fl, fl + ngeom) {
    for (int i = 0; i < count; ++i) workers.emplace_back(std::make_unique<Worker>(m));
    try {
      for (int i = 0; i < count; ++i) threads.emplace_back([this, i] { loop(i); });
    } catch (...) { stop(); throw; }
  }
  ~Context() { stop(); }
  void stop() {
    { std::lock_guard<std::mutex> guard(mutex); stopping = true; }
    ready.notify_all();
    for (auto& t : threads) if (t.joinable()) t.join();
  }
  void loop(size_t worker) {
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> guard(mutex);
      ready.wait(guard, [&] { return stopping || generation != seen; });
      if (stopping) return;
      seen = generation;
      guard.unlock();
      try {
        for (;;) {
          size_t frame = next_frame.fetch_add(1);
          if (frame >= frame_count) break;
          job(frame, *workers[worker]);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!failure) failure = std::current_exception();
        next_frame.store(frame_count);
      }
      guard.lock();
      if (--remaining == 0) finished.notify_one();
    }
  }
  void run(size_t frames, std::function<void(size_t, Worker&)> task) {
    callbacks_supported(allowed_timer);
    std::unique_lock<std::mutex> guard(mutex);
    frame_count = frames; next_frame.store(0); failure = nullptr;
    job = std::move(task); remaining = threads.size(); ++generation;
    ready.notify_all();
    finished.wait(guard, [&] { return remaining == 0; });
    job = {};
    if (failure) std::rethrow_exception(failure);
  }
  void collect(Worker& w, bool ground, std::vector<Row>& out) {
    const double threshold = ground ? ground_threshold : self_threshold;
    const auto& robot = ground ? ground_robot : self_mask;
    if (std::none_of(robot.begin(), robot.end(), [](uint8_t value) { return value != 0; }) ||
        (ground && std::none_of(floor_mask.begin(), floor_mask.end(), [](uint8_t value) { return value != 0; }))) return;
    std::set<std::pair<int, int>> candidates;
    for (int i = 0; i < ngeom; ++i) w.model->geom_margin[i] = std::max(w.saved_margin[i], threshold);
    const int old_contacts = w.data->warning[mjWARN_CONTACTFULL].number;
    mj_collision(w.model, w.data);
    std::copy(w.saved_margin.begin(), w.saved_margin.end(), w.model->geom_margin);
    if (w.data->warning[mjWARN_CONTACTFULL].number != old_contacts)
      throw std::runtime_error("MuJoCo contact arena overflow; no truncated geometry is returned");
    for (int i = 0; i < w.data->ncon; ++i) {
      int a = w.data->contact[i].geom[0], b = w.data->contact[i].geom[1];
      if (a < 0 || b < 0) continue;
      if (ground) {
        if (robot[a] && floor_mask[b]) candidates.emplace(a, b);
        else if (robot[b] && floor_mask[a]) candidates.emplace(b, a);
      } else if (robot[a] && robot[b]) candidates.emplace(std::min(a, b), std::max(a, b));
    }
    for (auto pair : candidates) {
      Row row{pair.first, pair.second, uint8_t(ground), 0., ground ? ground_margin : self_margin, {}};
      bool swapped = w.model->geom_type[row.a] > w.model->geom_type[row.b];
      row.distance = mj_geomDistance(w.model, w.data, swapped ? row.b : row.a,
                                    swapped ? row.a : row.b, threshold, row.witness.data());
      if (swapped) for (int k = 0; k < 3; ++k) std::swap(row.witness[k], row.witness[k + 3]);
      if (row.distance <= threshold) out.push_back(row);
    }
  }
  std::pair<int, int> box_plane(int a, int b, uint8_t kind) const {
    if (!kind) return {-1, -1};
    const auto* m = workers[0]->model;
    for (int i = 0; i < 2; ++i) {
      if (m->geom_type[a] == mjGEOM_BOX && m->geom_type[b] == mjGEOM_PLANE && m->geom_bodyid[b] == 0)
        return {a, b};
      std::swap(a, b);
    }
    return {-1, -1};
  }
  void jacobian(Worker& w, int a, int b, double distance, const double* witness, double* output) {
    double normal[3] = {witness[0] - witness[3], witness[1] - witness[4], witness[2] - witness[5]};
    const double norm = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    const double sign = distance >= 0. ? 1. : -1.;
    if (norm > 1e-12) for (double& v : normal) v = sign * v / norm;
    else { normal[0] = normal[1] = 0.; normal[2] = floor_label[b] ? sign : floor_label[a] ? -sign : 0.; }
    mj_jac(w.model, w.data, w.jp1.data(), w.jr1.data(), witness, w.model->geom_bodyid[a]);
    mj_jac(w.model, w.data, w.jp2.data(), w.jr2.data(), witness + 3, w.model->geom_bodyid[b]);
    for (int j = 0; j < nv; ++j) {
      output[j] = 0.;
      for (int k = 0; k < 3; ++k) output[j] += normal[k] * (w.jp1[k*nv+j] - w.jp2[k*nv+j]);
    }
  }
  void corners(Worker& w, int box, int plane, double margin, double* A, double* b) {
    const double* rotation = w.data->geom_xmat + 9*box;
    const double* plane_rotation = w.data->geom_xmat + 9*plane;
    const double* center = w.data->geom_xpos + 3*box;
    const double* origin = w.data->geom_xpos + 3*plane;
    const double* size = w.model->geom_size + 3*box;
    const double normal[3] = {plane_rotation[2], plane_rotation[5], plane_rotation[8]};
    mj_jac(w.model, w.data, w.jp1.data(), w.jr1.data(), center, w.model->geom_bodyid[box]);
    for (int c = 0; c < 8; ++c) {
      double offset[3] = {};
      for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j) offset[k] += rotation[3*k+j] * size[j] * ((c & (1 << (2-j))) ? 1. : -1.);
      const double cross[3] = {offset[1]*normal[2]-offset[2]*normal[1],
                               offset[2]*normal[0]-offset[0]*normal[2],
                               offset[0]*normal[1]-offset[1]*normal[0]};
      b[c] = -margin - safety;
      for (int k = 0; k < 3; ++k) b[c] += (center[k] + offset[k] - origin[k]) * normal[k];
      for (int j = 0; j < nv; ++j) {
        A[c*nv+j] = 0.;
        for (int k = 0; k < 3; ++k) A[c*nv+j] -= normal[k]*w.jp1[k*nv+j] + cross[k]*w.jr1[k*nv+j];
      }
    }
  }
};
template <class F> int checked(F function) {
  try { last_error.clear(); function(); return 0; }
  catch (const Unsupported& e) { last_error = e.what(); return 2; }
  catch (const std::exception& e) { last_error = e.what(); return 1; }
  catch (...) { last_error = "Unknown native geometry failure"; return 1; }
}
}

extern "C" {
struct NativeGeometryView {
  int64_t frames, rows, corners, nv;
  const int64_t *offsets, *corner_rows;
  const int32_t* pairs;
  const uint8_t* kinds;
  const double *distances, *margins, *witnesses, *jacobians, *corner_A, *corner_b;
};
const char* umr_geometry_error() { return last_error.c_str(); }
int umr_geometry_create(uintptr_t model_address, int expected_version, int count, int self_enabled,
    const uint8_t* sm, const uint8_t* gr, const uint8_t* fm, const uint8_t* fl,
    double st, double gt, double s_margin, double g_margin, double safety, uintptr_t allowed_timer, void** output) {
  return checked([&] {
    *output = nullptr;
    if (mj_version() != expected_version || mj_version() != mjVERSION_HEADER)
      throw Unsupported("MuJoCo header/runtime/Python version mismatch");
    callbacks_supported(allowed_timer);
    const auto* model = reinterpret_cast<const mjModel*>(model_address);
    if (!model || (count != 4 && count != 8)) throw std::invalid_argument("Invalid model or worker count");
    if (model->nplugin) throw Unsupported("MuJoCo plugins require the Python geometry backend");
    *output = new Context(model, count, self_enabled, sm, gr, fm, fl, st, gt, s_margin, g_margin, safety, allowed_timer);
  });
}
void umr_geometry_destroy(void* context) { delete static_cast<Context*>(context); }
int umr_geometry_measure(void* context, const double* poses, int64_t frames, void** output) {
  return checked([&] {
    *output = nullptr;
    auto& c = *static_cast<Context*>(context);
    if (frames < 0) throw std::invalid_argument("Negative frame count");
    std::vector<std::vector<Row>> rows(frames);
    c.run(frames, [&](size_t frame, Worker& worker) {
      worker.forward(poses + frame*c.nq);
      if (c.self_enabled) c.collect(worker, false, rows[frame]);
      c.collect(worker, true, rows[frame]);
    });
    auto result = std::make_unique<Result>();
    result->frames = frames; result->nv = c.nv; result->offsets.push_back(0);
    for (auto& frame : rows) {
      for (auto& row : frame) {
        result->pairs.insert(result->pairs.end(), {row.a, row.b}); result->kinds.push_back(row.kind);
        result->distances.push_back(row.distance); result->margins.push_back(row.margin);
        result->witnesses.insert(result->witnesses.end(), row.witness.begin(), row.witness.end());
      }
      result->offsets.push_back(result->distances.size());
    }
    result->rows = result->distances.size(); *output = result.release();
  });
}
int umr_geometry_linearize(void* context, const double* poses, int64_t frames, int64_t rows,
    const int64_t* offsets, const int32_t* pairs, const uint8_t* kinds, const double* distances,
    const double* margins, const double* witnesses, void** output) {
  return checked([&] {
    *output = nullptr;
    auto& c = *static_cast<Context*>(context);
    auto result = std::make_unique<Result>();
    result->frames = frames; result->rows = rows; result->nv = c.nv;
    result->jacobians.resize(rows*c.nv);
    std::vector<int64_t> corner_index(rows, -1);
    for (int64_t row = 0; row < rows; ++row) {
      if (c.box_plane(pairs[2*row], pairs[2*row+1], kinds[row]).first >= 0) {
        corner_index[row] = result->corner_rows.size(); result->corner_rows.push_back(row);
      }
    }
    result->corners = result->corner_rows.size();
    result->corner_A.resize(result->corners*8*c.nv); result->corner_b.resize(result->corners*8);
    c.run(frames, [&](size_t frame, Worker& worker) {
      worker.forward(poses + frame*c.nq);
      for (int64_t row = offsets[frame]; row < offsets[frame+1]; ++row) {
        c.jacobian(worker, pairs[2*row], pairs[2*row+1], distances[row], witnesses+6*row,
                   result->jacobians.data()+row*c.nv);
        if (corner_index[row] >= 0) {
          auto pair = c.box_plane(pairs[2*row], pairs[2*row+1], kinds[row]);
          c.corners(worker, pair.first, pair.second, margins[row],
                    result->corner_A.data()+corner_index[row]*8*c.nv,
                    result->corner_b.data()+corner_index[row]*8);
        }
      }
    });
    *output = result.release();
  });
}
void umr_geometry_view(void* opaque, NativeGeometryView* view) {
  const auto& r = *static_cast<Result*>(opaque);
  *view = {r.frames, r.rows, r.corners, r.nv, r.offsets.data(), r.corner_rows.data(), r.pairs.data(),
           r.kinds.data(), r.distances.data(), r.margins.data(), r.witnesses.data(), r.jacobians.data(),
           r.corner_A.data(), r.corner_b.data()};
}
void umr_geometry_release(void* result) { delete static_cast<Result*>(result); }
}
