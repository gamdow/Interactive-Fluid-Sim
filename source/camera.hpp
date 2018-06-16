#pragma once

#include <opencv2/opencv.hpp>
#include <ostream>
#include <thread>

#include "debug.hpp"
#include "component.hpp"
#include "renderer.hpp"
#include "data/resolution.cuh"
#include "data/render_quad.hpp"

struct Camera : public Component, public Renderable {
  cv::Mat const & data() const {return __data();}
  Resolution resolution() const {return __resolution();}
  Camera();
  virtual ~Camera() {}
private:
  virtual cv::Mat const & __data() const = 0;
  virtual Resolution __resolution() const = 0;
  virtual void __render(Resolution const & _window_res, float _mag, float2 _off);
  RenderQuad __quad;
};

struct CVCamera : public Debug<CVCamera>, public Camera {
  CVCamera(int _index, Resolution _res, float _fps);
  virtual ~CVCamera();
private:
  void capture();
  void setCameraProps(Resolution _res, float _fps);
  void reportFrameInfo(std::ostream & _out);
  virtual cv::Mat const & __data() const {return __frame;}
  virtual Resolution __resolution() const;
  cv::VideoCapture __camera;
  cv::Mat __frame;
  std::thread __capture;
  bool __continue;
};

struct NullCamera : public Debug<NullCamera>, public Camera {
  NullCamera(Resolution _res);
private:
  virtual cv::Mat const & __data() const {return __frame;}
  virtual Resolution __resolution() const {return __res;}
  cv::Mat __frame;
  Resolution __res;
};
