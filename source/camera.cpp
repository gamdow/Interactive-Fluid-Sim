#include "camera.h"

#include <string>
#include <cuda_runtime.h>

#include "debug.h"
#include "cuda/helper_cuda.h"
#include "renderer.h"

Camera::Camera(IRenderer & _renderer)
  : __renderTarget(_renderer.newTextureRenderTarget(GL_RGB, GL_BGR, GL_UNSIGNED_BYTE))
{
}

Camera::Camera(IRenderer & _renderer, Resolution const & _res, cv::Mat const & _mat)
  : __renderTarget(_renderer.newTextureRenderTarget(GL_RGB, GL_BGR, GL_UNSIGNED_BYTE))
  , __resolution(_res)
  , __frame(_mat)
{
}

void Camera::__render() {
  __renderTarget.bindTexture(__frame.cols, __frame.rows, __frame.data);
  __renderTarget.render();
}

NullCamera::NullCamera(IRenderer & _renderer, Resolution _res)
  : Camera(_renderer, _res, cv::Mat::zeros(_res.height, _res.width, CV_8UC3))
{
  format_out << "Null Camera:" << std::endl;
  float r = _res.height / 20;
  for(float k = -2.0f; k < 3.0f; k += 1.0f) {
    float odd = int(k) % 2;
    for(float l = -1 - odd / 2; l < 2 + odd / 2; l += 1.0f) {
      float2 center = make_float2(4 * r * k + _res.width / 2, 4 * r * l + _res.height / 2);
      for(int i = 0; i < _res.width; ++i) {
        for(int j = 0; j < _res.height; ++j) {
          if((center.y - j) * (center.y - j) + (center.x - i) * (center.x - i) < r * r) {
            cv::Vec3b & e = frame().at<cv::Vec3b>(j, i);
            e[0] = 255u;
            e[1] = 255u;
            e[2] = 255u;
          }
        }
      }
    }
  }
  for(int i = 0; i < 10; ++i) {
    for(int j = 0; j < 10; ++j) {
      {
        cv::Vec3b & e = frame().at<cv::Vec3b>(j, i);
        e[0] = 255u;
        e[1] = 255u;
        e[2] = 255u;
      }
      {
        cv::Vec3b & e = frame().at<cv::Vec3b>(_res.height - j, _res.width - i);
        e[0] = 255u;
        e[1] = 255u;
        e[2] = 255u;
      }
    }
  }
  for(int i = 0; i < 10; ++i) {
    for(int j = 0; j < 10; ++j) {
      cv::Vec3b & e = frame().at<cv::Vec3b>(j, i);
      e[0] = 255u;
      e[1] = 255u;
      e[2] = 255u;
    }
  }
}

CVCamera::CVCamera(IRenderer & _renderer, int _index, Resolution _res, float _fps)
  : Camera(_renderer)
  , __camera(_index)
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
