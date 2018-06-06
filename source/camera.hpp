#pragma once

#include <opencv2/opencv.hpp>
#include <ostream>
#include <thread>

#include "component.hpp"
#include "resolution.cuh"

struct Camera : public Component {
  cv::Mat const & data() const {return __data();}
  Resolution resolution() const {return __resolution();}
private:
  virtual cv::Mat const & __data() const = 0;
  virtual Resolution __resolution() const = 0;
};

struct CVCamera : public Camera {
  CVCamera(int _index, Resolution _res, float _fps);
private:
  void setCameraProps(Resolution _res, float _fps);
  void reportFrameInfo(std::ostream & _out);
  virtual cv::Mat const & __data() const {return __frame;}
  virtual Resolution __resolution() const;
  cv::VideoCapture __camera;
  cv::Mat __frame;
  std::thread __capture;
};

struct NullCamera : public Camera {
  NullCamera(Resolution _res);
private:
  virtual cv::Mat const & __data() const {return __frame;}
  virtual Resolution __resolution() const {return __res;}
  cv::Mat __frame;
  Resolution __res;
};
