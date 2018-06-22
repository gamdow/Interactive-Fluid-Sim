#include "simulation_wrapper.cuh"

#include "simulation.cuh"
#include "general.cuh"

SimulationWrapper::SimulationWrapper(OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx)
  : KernelWrapper(_block_config, _buffer_width)
  , __dx(_dx)
  , __rdx(make_float2(1.0f / _dx.x, 1.0f / _dx.y))
{
  Allocator alloc;
  __fluid_cells.resize(alloc, buffer_resolution().size);
  __divergence.resize(alloc, buffer_resolution().size);
  __pressure.resize(alloc, buffer_resolution().size);
  __velocity.resize(alloc, buffer_resolution().size);
  __smoke.resize(alloc, buffer_resolution().size);
  __f1_temp.resize(alloc, buffer_resolution().size);
  __f2_temp_texture.init(alloc, buffer_resolution());
  __f4_temp_texture.init(alloc, buffer_resolution());
}

void SimulationWrapper::copyToFluidCells(uchar3 const * _buffer, Resolution const & _in_res) {
  copy_to_array<<<grid(),block()>>>(__fluid_cells, buffer_resolution(), _buffer, _in_res);
}

void SimulationWrapper::advectVelocity(float _dt) {
  advect(__velocity, __velocity, _dt);
}

void SimulationWrapper::calcDivergence() {
  calc_divergence<<<grid(),block()>>>(__divergence, __velocity, __fluid_cells, buffer_resolution(), __rdx);
}

void SimulationWrapper::pressureDecay() {
  pressure_decay<<<grid(),block()>>>(__pressure, __fluid_cells, buffer_resolution());
}

void SimulationWrapper::pressureSolveStep() {
  pressure_solve<<<grid(),block()>>>(__f1_temp, __pressure, __divergence, __fluid_cells, buffer_resolution(), __dx);
  swap(__f1_temp, __pressure);
}

void SimulationWrapper::subGradient() {
  sub_gradient<<<grid(),block()>>>(__velocity, __pressure, __fluid_cells, buffer_resolution(), __rdx);
}

void SimulationWrapper::enforceSlip() {
  enforce_slip<<<grid(),block()>>>(__velocity, __fluid_cells, buffer_resolution());
}

void SimulationWrapper::advectSmoke(float _dt) {
  __f4_temp_texture.copyFrom(__smoke, buffer_resolution());
  apply_advection<<<grid(),block()>>>((float4 *)__smoke, __f4_temp_texture.getObject(), __velocity, __fluid_cells, buffer_resolution(), _dt, __rdx);
}

void SimulationWrapper::advect(float2 * _out, float2 const * _in, float _dt) {
  __f2_temp_texture.copyFrom(_in, buffer_resolution());
  advect_velocity<<<grid(),block()>>>(_out, __f2_temp_texture.getObject(), buffer_resolution(), _dt, __rdx);
}

BFECCSimulationWrapper::BFECCSimulationWrapper(OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx)
  : SimulationWrapper(_block_config, _buffer_width, _dx)
{
  Allocator alloc;
  __f2_temp.resize(alloc, buffer_resolution().size);
}

void BFECCSimulationWrapper::advectVelocity(float _dt) {
  advect(__f2_temp, __velocity, _dt);
  advect(__f2_temp, __f2_temp, -_dt);
  sum_arrays<<<grid(),block()>>>(__velocity, 1.5f, __velocity, -.5f, __f2_temp, buffer_resolution());
  advect(__velocity, __velocity, _dt);
}

LBFECCSimulationWrapper::LBFECCSimulationWrapper(OptimalBlockConfig const & _block_config, int _buffer_width, float2 _dx)
  : SimulationWrapper(_block_config, _buffer_width, _dx)
{
  Allocator alloc;
  __f2_tempA.resize(alloc, buffer_resolution().size);
  __f2_tempB.resize(alloc, buffer_resolution().size);
  __f2_tempC.resize(alloc, buffer_resolution().size);
}

void LBFECCSimulationWrapper::advectVelocity(float _dt) {
  advect(__f2_tempA, __velocity, _dt);
  advect(__f2_tempA, __f2_tempA, -_dt);
  sum_arrays<<<grid(),block()>>>(__f2_tempB, .5f, __velocity, -.5f, __f2_tempA, buffer_resolution());
  sum_arrays<<<grid(),block()>>>(__f2_tempA, 1.0f, __velocity, 1.0f, __f2_tempB, buffer_resolution());
  advect(__f2_tempA, __f2_tempA, _dt);
  advect(__f2_tempA, __f2_tempA, -_dt);
  sum_arrays<<<grid(),block()>>>(__f2_tempA, 1.0f, __f2_tempA, 1.0f, __f2_tempB, buffer_resolution());
  sum_arrays<<<grid(),block()>>>(__f2_tempA, 1.0f, __velocity, -1.0f, __f2_tempA, buffer_resolution());
  limit_advection<<<grid(),block()>>>(__f2_tempC, __f2_tempB, __f2_tempA, buffer_resolution());
  sum_arrays<<<grid(),block()>>>(__velocity, 1.0f, __velocity, 1.0f, __f2_tempC, buffer_resolution());
  advect(__velocity, __velocity, _dt);
}
