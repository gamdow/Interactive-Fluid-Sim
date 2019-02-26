#include "simulation.h"

#include <cuda_runtime.h>

#include "../data/resolution.h"
#include "../interface/enums.h"

__global__ void advect_velocity(float2 * o_velocity, cudaTextureObject_t _velocityObj, Resolution _buffer_res, float _dt, float2 _rdx);
__global__ void limit_advection(float2 * o_e, float2 * _e1, float2 * _e2, Resolution _buffer_res);
template<class T> __global__ void apply_advection(T * o_data, cudaTextureObject_t _dataObj, float2 const * _velocity, float const * _fluid, Resolution _buffer_res, float _dt, float2 _rdx);
__global__ void calc_divergence(float * o_divergence, float2 const * _velocity, float const * _fluid, Resolution _buffer_res, float2 _rdx);
__global__ void pressure_decay(float * io_pressure, float const * _fluid, Resolution _buffer_res);
__global__ void pressure_solve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, Resolution _buffer_res, float2 _dx);
__global__ void sub_gradient(float2 * io_velocity, float const * _pressure, float const * _fluid, Resolution _buffer_res, float2 _rdx);
__global__ void enforce_slip(float2 * io_velocity, float const * _fluid, Resolution _buffer_res);
__global__ void sum_arrays(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2, Resolution _buffer_res);
__global__ void apply_smoke(float4 * o_array, Resolution _buffer_res, int _flow_direction);
__global__ void inject_velocity_and_apply_boundary(float2 * o_velocity, float * o_fluid, Resolution _buffer_res, int _flow_direction, float _velocity_setting);

SimulationWrapper::SimulationWrapper(OptimalBlockConfig const & _block_config, float2 _dx)
  : __config(_block_config)
  , __dx(_dx)
  , __rdx(make_float2(1.0f / _dx.x, 1.0f / _dx.y))
{
  format_out << "Constructing Simulation Kernel Buffers:" << std::endl;
  OutputIndent indent;
  Allocator alloc;
  __fluid_cells.resize(alloc, __config.resolution.total_size());
  __divergence.resize(alloc, __config.resolution.total_size());
  __pressure.resize(alloc, __config.resolution.total_size());
  __velocity.resize(alloc, __config.resolution.total_size());
  __smoke.resize(alloc, __config.resolution.total_size());
  __f1_temp.resize(alloc, __config.resolution.total_size());
  __f2_temp_texture.init(alloc, __config.resolution);
  __f4_temp_texture.init(alloc, __config.resolution);
}

void SimulationWrapper::advectVelocity(float _dt) {
  advect(__velocity, __velocity, _dt);
}

void SimulationWrapper::calcDivergence() {
  calc_divergence<<<__config.inner_grid,__config.block>>>(__divergence, __velocity, __fluid_cells, __config.resolution, __rdx);
}

void SimulationWrapper::pressureDecay() {
  pressure_decay<<<__config.inner_grid,__config.block>>>(__pressure, __fluid_cells, __config.resolution);
}

void SimulationWrapper::pressureSolveStep() {
  pressure_solve<<<__config.inner_grid,__config.block>>>(__f1_temp, __pressure, __divergence, __fluid_cells, __config.resolution, __dx);
  swap(__f1_temp, __pressure);
}

void SimulationWrapper::subGradient() {
  sub_gradient<<<__config.inner_grid,__config.block>>>(__velocity, __pressure, __fluid_cells, __config.resolution, __rdx);
}

void SimulationWrapper::enforceSlip() {
  enforce_slip<<<__config.inner_grid,__config.block>>>(__velocity, __fluid_cells, __config.resolution);
}

void SimulationWrapper::advectSmoke(float _dt) {
  __f4_temp_texture.copyFrom(__smoke, __config.resolution);
  apply_advection<<<__config.inner_grid,__config.block>>>((float4 *)__smoke, __f4_temp_texture.getObject(), __velocity, __fluid_cells, __config.resolution, _dt, __rdx);
}

void SimulationWrapper::applySmoke(int _flow_direction) {
  apply_smoke<<<__config.inner_grid,__config.block>>>((float4 *)__smoke, __config.resolution, _flow_direction);
}

void SimulationWrapper::injectVelocityAndApplyBoundary(int _flow_direction, float _velocity_setting) {
  inject_velocity_and_apply_boundary<<<__config.buffered_grid,__config.block>>>(__velocity, __fluid_cells, __config.resolution, _flow_direction, _velocity_setting);
}

void SimulationWrapper::advect(float2 * _out, float2 const * _in, float _dt) {
  __f2_temp_texture.copyFrom(_in, __config.resolution);
  advect_velocity<<<__config.inner_grid,__config.block>>>(_out, __f2_temp_texture.getObject(), __config.resolution, _dt, __rdx);
}

BFECCSimulationWrapper::BFECCSimulationWrapper(OptimalBlockConfig const & _block_config, float2 _dx)
  : SimulationWrapper(_block_config, _dx)
{
  format_out << "Constructing Additional BFECC Simulation Kernel Buffers:" << std::endl;
  Allocator alloc;
  __f2_temp.resize(alloc, __config.resolution.total_size());
}

void BFECCSimulationWrapper::advectVelocity(float _dt) {
  advect(__f2_temp, __velocity, _dt);
  advect(__f2_temp, __f2_temp, -_dt);
  sum_arrays<<<__config.inner_grid,__config.block>>>(__velocity, 1.5f, __velocity, -.5f, __f2_temp, __config.resolution);
  advect(__velocity, __velocity, _dt);
}

LBFECCSimulationWrapper::LBFECCSimulationWrapper(OptimalBlockConfig const & _block_config, float2 _dx)
  : SimulationWrapper(_block_config, _dx)
{
  format_out << "Constructing Additional LBFECC Simulation Kernel Buffers:" << std::endl;
  OutputIndent indent;
  Allocator alloc;
  __f2_tempA.resize(alloc, __config.resolution.total_size());
  __f2_tempB.resize(alloc, __config.resolution.total_size());
  __f2_tempC.resize(alloc, __config.resolution.total_size());
}

void LBFECCSimulationWrapper::advectVelocity(float _dt) {
  advect(__f2_tempA, __velocity, _dt);
  advect(__f2_tempA, __f2_tempA, -_dt);
  sum_arrays<<<__config.inner_grid,__config.block>>>(__f2_tempB, .5f, __velocity, -.5f, __f2_tempA, __config.resolution);
  sum_arrays<<<__config.inner_grid,__config.block>>>(__f2_tempA, 1.0f, __velocity, 1.0f, __f2_tempB, __config.resolution);
  advect(__f2_tempA, __f2_tempA, _dt);
  advect(__f2_tempA, __f2_tempA, -_dt);
  sum_arrays<<<__config.inner_grid,__config.block>>>(__f2_tempA, 1.0f, __f2_tempA, 1.0f, __f2_tempB, __config.resolution);
  sum_arrays<<<__config.inner_grid,__config.block>>>(__f2_tempA, 1.0f, __velocity, -1.0f, __f2_tempA, __config.resolution);
  limit_advection<<<__config.inner_grid,__config.block>>>(__f2_tempC, __f2_tempB, __f2_tempA, __config.resolution);
  sum_arrays<<<__config.inner_grid,__config.block>>>(__velocity, 1.0f, __velocity, 1.0f, __f2_tempC, __config.resolution);
  advect(__velocity, __velocity, _dt);
}

template<typename T>
__device__
inline T lerp(T a, T b, T l) {
  //return (1. - l) * a + l * b;
  return fma(l, b, fma(-l, a, a));
}

__global__ void advect_velocity(float2 * o_velocity, cudaTextureObject_t _velocityObj, Resolution _buffer_res, float _dt, float2 _rdx) {
  float s = (float)_buffer_res.x() + 0.5f;
  float t = (float)_buffer_res.y() + 0.5f;
  float2 pos = make_float2(s, t) - _dt * _rdx * tex2D<float2>(_velocityObj, s, t);
  o_velocity[_buffer_res.idx()] = tex2D<float2>(_velocityObj, pos.x, pos.y);
}

template<typename T>
__global__ void apply_advection(T * o_data, cudaTextureObject_t _dataObj, float2 const * _velocity, float const * _fluid, Resolution _buffer_res, float _dt, float2 _rdx) {
  if(_fluid[_buffer_res.idx()] > 0.0f) {
    float2 pos = make_float2(_buffer_res.x() + 0.5f, _buffer_res.y() + 0.5f) - _dt * _rdx * _velocity[_buffer_res.idx()];
    o_data[_buffer_res.idx()] = tex2D<T>(_dataObj, pos.x, pos.y);
  } else {
    o_data[_buffer_res.idx()] *= 0.9f;
  }
}

__global__ void calc_divergence(float * o_divergence, float2 const * _velocity, float const * _fluid, Resolution _buffer_res, float2 _rdx) {
  int4 const stencil = _buffer_res.stencil();
  o_divergence[_buffer_res.idx()] = (_velocity[stencil.x].x - _velocity[stencil.y].x) * (_rdx.x / 2.0f) + (_velocity[stencil.z].y - _velocity[stencil.w].y) * (_rdx.y / 2.0f);
}

__global__ void pressure_decay(float * io_pressure, float const * _fluid, Resolution _buffer_res) {
  int const idx = _buffer_res.idx();
  io_pressure[idx] *= _fluid[idx] * 0.1f + 0.9f;
}

__global__ void pressure_solve(float * o_pressure, float const * _pressure, float const * _divergence, float const * _fluid, Resolution _buffer_res, float2 _dx) {
  int const idx = _buffer_res.idx();
  int4 const stencil = _buffer_res.stencil();
  float pR = lerp(_pressure[idx], _pressure[stencil.x], _fluid[stencil.x]);
  float pL = lerp(_pressure[idx], _pressure[stencil.y], _fluid[stencil.y]);
  float pU = lerp(_pressure[idx], _pressure[stencil.z], _fluid[stencil.z]);
  float pD = lerp(_pressure[idx], _pressure[stencil.w], _fluid[stencil.w]);
  o_pressure[idx] = (1.0f / 4.0f) * (pR + pL + pU + pD
    - _divergence[idx] * _dx.x * _dx.y);
}

__global__ void sub_gradient(float2 * io_velocity, float const * _pressure, float const * _fluid, Resolution _buffer_res, float2 _rdx) {
  int const idx = _buffer_res.idx();
  int4 const stencil = _buffer_res.stencil();
  float pR = lerp(_pressure[idx], _pressure[stencil.x], _fluid[stencil.x]);
  float pL = lerp(_pressure[idx], _pressure[stencil.y], _fluid[stencil.y]);
  float pU = lerp(_pressure[idx], _pressure[stencil.z], _fluid[stencil.z]);
  float pD = lerp(_pressure[idx], _pressure[stencil.w], _fluid[stencil.w]);
  io_velocity[idx] -= _fluid[idx] * (_rdx / 2.0f) * make_float2(pR - pL, pU - pD);
}

__global__ void enforce_slip(float2 * io_velocity, float const * _fluid, Resolution _buffer_res) {
  int const idx = _buffer_res.idx();
  int4 const stencil = _buffer_res.stencil();
  if(_fluid[idx] > 0.0f) {
    float xvel = _fluid[stencil.x] * _fluid[stencil.y] == 0.0f
      ? ((1.f - _fluid[stencil.x]) * io_velocity[stencil.x].x + (1.f - _fluid[stencil.y]) * io_velocity[stencil.y].x) / (2.f - _fluid[stencil.x] - _fluid[stencil.y])
      : io_velocity[idx].x;
    float yvel = _fluid[stencil.z] * _fluid[stencil.w] == 0.0f
      ? ((1.f - _fluid[stencil.z]) * io_velocity[stencil.z].y + (1.f - _fluid[stencil.w]) * io_velocity[stencil.w].y) / (2.f - _fluid[stencil.z] - _fluid[stencil.w])
      : io_velocity[idx].y;
    io_velocity[idx] = make_float2(xvel, yvel);
  } else {
    io_velocity[idx] = make_float2(0.0f, 0.0f);
  }
}

__device__ inline float minmod(float a, float b) {
  return a * b > 0
    ? (a > 0 ? fminf(a, b) : fmaxf(a, b))
    : 0;
}

__device__ inline float2 minmod2(float2 a, float2 b) {
  return make_float2(minmod(a.x, b.x), minmod(a.y, b.y));
}

__device__ inline float2 limit_select(float2 * _e1, float2 * _e2, int i, int j) {
  return make_float2(_e2[j].x * _e2[j].x > _e1[j].x * _e1[j].x ? _e1[i].x : _e1[j].x, _e2[j].y * _e2[j].y > _e1[j].y * _e1[j].y ? _e1[i].y : _e1[j].y);
}

__global__ void limit_advection(float2 * o_e, float2 * _e1, float2 * _e2, Resolution _buffer_res) {
  int4 const stencil = _buffer_res.stencil();
  o_e[_buffer_res.idx()] = minmod2(
    minmod2(limit_select(_e1, _e2, _buffer_res.idx(), stencil.x), limit_select(_e1, _e2, _buffer_res.idx(), stencil.y)),
    minmod2(limit_select(_e1, _e2, _buffer_res.idx(), stencil.z), limit_select(_e1, _e2, _buffer_res.idx(), stencil.w)));
}

__global__ void sum_arrays(float2 * o_array, float _c1, float2 const * _array1, float _c2, float2 const * _array2, Resolution _buffer_res) {
  int const idx = _buffer_res.idx();
  o_array[idx] = _c1 * _array1[idx] + _c2 * _array2[idx];
}

__global__ void apply_smoke(float4 * o_array, Resolution _buffer_res, int _flow_direction) {
  int x_min, x_max, y_min, y_max;
  switch (_flow_direction) {
    case FlowDirection::LEFT_TO_RIGHT:
      x_min = _buffer_res.width.begin();
      x_max = _buffer_res.width.begin_inner() + _buffer_res.width.buffer;
      break;
    case FlowDirection::RIGHT_TO_LEFT:
      x_min = _buffer_res.width.end_inner() - _buffer_res.width.buffer;
      x_max = _buffer_res.width.end();
      break;
    case FlowDirection::TOP_TO_BOTTOM:
    case FlowDirection::BOTTOM_TO_TOP:
      x_min = _buffer_res.width.begin_inner();
      x_max = _buffer_res.width.end_inner();
      break;
  }
  switch (_flow_direction) {
    case FlowDirection::LEFT_TO_RIGHT:
    case FlowDirection::RIGHT_TO_LEFT:
      y_min = _buffer_res.height.begin_inner();
      y_max = _buffer_res.height.end_inner();
      break;
    case FlowDirection::TOP_TO_BOTTOM:
      y_min = _buffer_res.height.begin();
      y_max = _buffer_res.height.begin_inner() + _buffer_res.height.buffer;
      break;
    case FlowDirection::BOTTOM_TO_TOP:
      y_min = _buffer_res.height.end_inner() - _buffer_res.height.buffer;
      y_max = _buffer_res.height.end();
      break;
  }

  if(_buffer_res.x() >= x_min && _buffer_res.x() < x_max && _buffer_res.y() >= y_min && _buffer_res.y() < y_max) {
    int z = 0;
    int const width = _buffer_res.height / 20;
    switch (_flow_direction) {
      case FlowDirection::LEFT_TO_RIGHT:
      case FlowDirection::RIGHT_TO_LEFT:
        z = _buffer_res.y();
        break;
      case FlowDirection::TOP_TO_BOTTOM:
      case FlowDirection::BOTTOM_TO_TOP:
        z = _buffer_res.x();
        break;
    }
    int c = (z / width) % 4;
    o_array[_buffer_res.idx()] = make_float4(
      c == 0 ? 1.0f : 0.f,
      c == 1 ? 1.0f : 0.f,
      c == 2 ? 1.0f : 0.f,
      c == 3 ? 1.0f : 0.f
    ) * powf(sinf(z * 3.14159f * (1.0f / width)), 2) * 1.5f;
  }
}

__global__ void inject_velocity_and_apply_boundary(float2 * o_velocity, float * o_fluid, Resolution _buffer_res, int _flow_direction, float _velocity_setting) {
  int const idx = _buffer_res.buffer_idx();
  switch (_flow_direction) {
    case FlowDirection::LEFT_TO_RIGHT:
    case FlowDirection::RIGHT_TO_LEFT:
      if(_buffer_res.j() < _buffer_res.height.begin_inner() || _buffer_res.j() >= _buffer_res.height.end_inner()) {
        o_fluid[idx] = 0.0f;
        o_velocity[idx] = make_float2(0.f, 0.f);
      } else if(_buffer_res.i() < _buffer_res.width.begin_inner() + _buffer_res.width.buffer || _buffer_res.i() >= _buffer_res.width.end_inner() - _buffer_res.width.buffer) {
        o_velocity[idx] = make_float2((_flow_direction == FlowDirection::LEFT_TO_RIGHT ? 1.0f : -1.0f) * _velocity_setting, 0.f);
      }
      break;
    case FlowDirection::TOP_TO_BOTTOM:
    case FlowDirection::BOTTOM_TO_TOP:
      if(_buffer_res.i() < _buffer_res.width.begin_inner() || _buffer_res.i() >= _buffer_res.width.end_inner()) {
        o_fluid[idx] = 0.0f;
        o_velocity[idx] = make_float2(0.f, 0.f);
      } else if(_buffer_res.j() < _buffer_res.height.begin_inner() + _buffer_res.height.buffer || _buffer_res.j() >= _buffer_res.height.end_inner() - _buffer_res.height.buffer) {
        o_velocity[idx] = make_float2(0.f, (_flow_direction == FlowDirection::TOP_TO_BOTTOM ? 1.0f : -1.0f) * _velocity_setting);
      }
      break;
  }
}
