#pragma once

dim3 const BLOCK_SIZE(32u, 18u);
static int2 const RESOLUTION = make_int2(BLOCK_SIZE.x * 20, BLOCK_SIZE.y * 20);
static int const BUFFER = 1u;
float2 const LENGTH = {1.6f, 0.9f};
float const FRAME_RATE = 60.0f;
