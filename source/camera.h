#pragma once

#include <thread>
#include <opencv2/opencv.hpp>

#include "component.h"
#include "data/resolution.h"
#include "data/managed_array.h"

struct Camera : public Component {
  virtual ~Camera() {}
  ArrayStructConst<uchar3> frameData() const {return ArrayStructConst<uchar3>(data(), __resolution);}
protected:
  Camera() {}
  Camera(Resolution const & _res, cv::Mat const & _mat);
  Resolution & resolution() {return __resolution;}
  cv::Mat & frame() {return __frame;}
private:
  uchar3 const * data() const {return reinterpret_cast<uchar3 const *>(__frame.data);}
  Resolution __resolution;
  cv::Mat __frame;
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
