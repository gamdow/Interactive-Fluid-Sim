#pragma once

#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <SDL2/SDL.h>

#include "component.h"
#include "data/resolution.h"
#include "data/managed_array.h"

struct Camera : public Component {
  virtual ~Camera() {}
  ArrayStructConst<uchar3> frameData() const {return ArrayStructConst<uchar3>(reinterpret_cast<uchar3 const *>(__frame.data), __resolution);}
  ArrayStructConst<float2> flowData() const {return ArrayStructConst<float2>(reinterpret_cast<float2 const *>(__flow.data), __resolution);}
  float & time_delta() {return __dt;}
protected:
  Camera() {}
  Camera(Resolution const & _res, cv::Mat const & _mat);
  Resolution & resolution() {return __resolution;}
  cv::Mat & frame() {return __frame;}
  cv::Mat & lastFrame() {return __last_frame;}
  cv::Mat & flow() {return __flow;}
private:
  uchar3 const * data() const {return reinterpret_cast<uchar3 const *>(__frame.data);}
  Resolution __resolution;
  cv::Mat __frame;
  cv::Mat __last_frame;
  cv::Mat __flow;
  float __dt;
};

struct NullCamera : public Camera {
  NullCamera(Resolution _res);
};

struct CVCamera : public Camera {
  CVCamera(bool _optical_flow, int _index, Resolution _res, float _fps);
private:
  struct Thread {
    Thread(bool _optical_flow) : OPTICAL_FLOW(_optical_flow), __continue(false) {}
    ~Thread() {stop();}
    void start(cv::VideoCapture & _cam, cv::Mat & _frame, cv::Mat & _last_frame, cv::Mat & _flow, float & _dt);
    void stop();
  private:
    void capture(cv::VideoCapture & _cam, cv::Mat & _frame, cv::Mat & _last_frame, cv::Mat & _flow, float & _dt);
    bool const OPTICAL_FLOW;
    bool __continue;
    Uint32 __last_ticks;
    std::thread __capture;
  };
  void setCameraProps(Resolution _res, float _fps);
  void reportFrameInfo();
  cv::VideoCapture __camera;
  Thread __capture;
};
