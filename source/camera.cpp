#include "camera.hpp"

#include <string>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

Camera::Camera()
  : __quad(GL_RGB, GL_BGR, GL_UNSIGNED_BYTE)
{

}

void Camera::__render(Resolution const & _window_res, float _mag, float2 _off) {
  __quad.bindTexture(__data());
  __quad.render(__resolution(), _window_res, _mag, _off);
}

CVCamera::CVCamera(int _index, Resolution _res, float _fps)
  : Debug<CVCamera>("OpenCV Camera:")
  , __camera(_index)
{
  if(!__camera.isOpened()) {
    std::stringstream error; error << "\tCannot open video camera (id:" << _index << ")";
    throwFailure(error.str());
  }
  std::cout << "\tVideo camera found (id:" << _index << ")" << std::endl;
  setCameraProps(_res, _fps);
  if(!__camera.read(__frame)) {
    std::stringstream error; error << "\tCould not capture frame";
    throwFailure(error.str());
  }
  reportFrameInfo(std::cout);
  __continue = true;
  __capture = std::thread(&CVCamera::capture, this);
}

CVCamera::~CVCamera() {
  __continue = false;
  __capture.join();
}

void CVCamera::capture() {
  while(__continue) {
    __camera.read(__frame);
  }
}

Resolution CVCamera::__resolution() const {
  return Resolution(__camera.get(cv::CAP_PROP_FRAME_WIDTH), __camera.get(cv::CAP_PROP_FRAME_HEIGHT));
}

void CVCamera::setCameraProps(Resolution _res, float _fps) {
  __camera.set(cv::CAP_PROP_FRAME_WIDTH, _res.width);
  __camera.set(cv::CAP_PROP_FRAME_HEIGHT, _res.height);
  __camera.set(cv::CAP_PROP_FPS, _fps);
  __resolution().print("\tResolution");
}

void CVCamera::reportFrameInfo(std::ostream & _out) {
  static char const * depth_names[] = {"8U", "8S", "16U", "16S", "32S", "32F", "64F"};
  _out << "\tDepth: " << depth_names[__frame.depth()] << std::endl;
  _out << "\tChannels: " << __frame.channels() << std::endl;
}

NullCamera::NullCamera(Resolution _res)
  : Debug<NullCamera>("Null Camera:")
  , __res(_res)
  , __frame(cv::Mat::zeros(_res.height, _res.width, CV_8UC3))
{
  for(int k = 2; k < 3; ++k) {
    int odd = k % 2;
    for(int l = 1; l < 2 + odd; ++l) {
      float2 center = make_float2(k * 80 - 2.5f * 80.f + _res.width / 2, l * 80 - 2.5f * 80.f + 100 - 40 * odd + _res.height / 2);
      for(int i = 0; i < _res.width; ++i) {
        for(int j = 0; j < _res.height; ++j) {
          if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 5000) {
            cv::Vec3b & e = __frame.at<cv::Vec3b>(j, i);
            e[0] = 255u;
            e[1] = 255u;
            e[2] = 255u;
          }
        }
      }
    }
  }
}
