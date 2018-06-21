#pragma once

#include <opencv2/opencv.hpp>
#include <thread>

#include "component.hpp"
#include "renderer.hpp"
#include "data/resolution.cuh"
#include "data/render_quad.hpp"

struct Camera : public Component, public Renderable {
  virtual ~Camera() {}
  Resolution const & resolution() const {return __resolution;}
  DeviceArray<uchar3> const & deviceArray() const {return __device;}
  void updateDeviceArray();
protected:
  Camera();
  Camera(Resolution const & _res, cv::Mat const & _mat);
  Resolution & resolution() {return __resolution;}
  cv::Mat & frame() {return __frame;}
private:
  virtual void __render(Resolution const & _window_res, float _mag, float2 _off);
  RenderQuad __quad;
  Resolution __resolution;
  cv::Mat __frame;
  DeviceArray<uchar3> __device;
};

struct NullCamera : public Camera {
  NullCamera(Resolution _res);
};

struct CVCamera : public Camera {
  CVCamera(int _index, Resolution _res, float _fps);
private:
  struct Thread {
    Thread() : __continue(false) {}
    ~Thread() {stop();}
    void start(cv::VideoCapture & _cam, cv::Mat & _frame);
    void stop();
  private:
    void capture(cv::VideoCapture & _cam, cv::Mat & _frame);
    bool __continue;
    std::thread __capture;
  };
  void setCameraProps(Resolution _res, float _fps);
  void reportFrameInfo();
  cv::VideoCapture __camera;
  Thread __capture;
};
