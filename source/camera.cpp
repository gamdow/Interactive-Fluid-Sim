#include "camera.hpp"

#include <string>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "debug.hpp"

Camera::Camera()
  : __quad(GL_RGB, GL_BGR, GL_UNSIGNED_BYTE)
{
}

Camera::Camera(Resolution const & _res, cv::Mat const & _mat)
  : __quad(GL_RGB, GL_BGR, GL_UNSIGNED_BYTE)
  , __resolution(_res)
  , __frame(_mat)
{
}

void Camera::updateDeviceArray() {
  if(__device.getSize() != __resolution.size) {
    __device.resize(Allocator(), __resolution.size);
  }
  checkCudaErrors(cudaMemcpy(__device, __frame.data, __device.getSizeBytes(), cudaMemcpyHostToDevice));
}

void Camera::__render(Resolution const & _window_res, float _mag, float2 _off) {
  __quad.bindTexture(__frame);
  __quad.render(__resolution, _window_res, _mag, _off);
}

NullCamera::NullCamera(Resolution _res)
  : Camera(_res, cv::Mat::zeros(_res.height, _res.width, CV_8UC3))
{
  format_out << "Null Camera:" << std::endl;
  for(int k = 2; k < 3; ++k) {
    int odd = k % 2;
    for(int l = 1; l < 2 + odd; ++l) {
      float2 center = make_float2(k * 80 - 2.5f * 80.f + _res.width / 2, l * 80 - 2.5f * 80.f + 100 - 40 * odd + _res.height / 2);
      for(int i = 0; i < _res.width; ++i) {
        for(int j = 0; j < _res.height; ++j) {
          if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < 5000) {
            cv::Vec3b & e = frame().at<cv::Vec3b>(j, i);
            e[0] = 255u;
            e[1] = 255u;
            e[2] = 255u;
          }
        }
      }
    }
  }
}

CVCamera::CVCamera(int _index, Resolution _res, float _fps)
  : __camera(_index)
{
  format_out << "OpenCV Camera:" << std::endl;
  OutputIndent indent;
  if(!__camera.isOpened()) {
    std::stringstream error; error << "Cannot open video camera (id:" << _index << ")";
    throwFailure(error.str());
  }
  format_out << "Video camera found (id:" << _index << ")" << std::endl;
  setCameraProps(_res, _fps);
  resolution() = Resolution(__camera.get(cv::CAP_PROP_FRAME_WIDTH), __camera.get(cv::CAP_PROP_FRAME_HEIGHT));
  resolution().print("Resolution");
  if(!__camera.read(frame())) {
    std::stringstream error; error << "Could not capture frame";
    throwFailure(error.str());
  }
  reportFrameInfo();
  __capture.start(__camera, frame());
}

void CVCamera::Thread::start(cv::VideoCapture & _cam, cv::Mat & _frame) {
  __continue = true;
  __capture = std::thread(&Thread::capture, this, std::ref(_cam), std::ref(_frame));
}

void CVCamera::Thread::stop() {
  if(__continue) {
    __continue = false;
    __capture.join();
  }
}

void CVCamera::Thread::capture(cv::VideoCapture & _cam, cv::Mat & _frame) {
  while(__continue) {
    _cam.read(_frame);
  }
}

void CVCamera::setCameraProps(Resolution _res, float _fps) {
  __camera.set(cv::CAP_PROP_FRAME_WIDTH, _res.width);
  __camera.set(cv::CAP_PROP_FRAME_HEIGHT, _res.height);
  __camera.set(cv::CAP_PROP_FPS, _fps);
}

void CVCamera::reportFrameInfo() {
  static char const * depth_names[] = {"8U", "8S", "16U", "16S", "32S", "32F", "64F"};
  format_out << "Depth: " << depth_names[frame().depth()] << std::endl;
  format_out << "Channels: " << frame().channels() << std::endl;
}
