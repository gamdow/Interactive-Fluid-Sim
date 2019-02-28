#include "camera.h"

#include <string>
#include <cuda_runtime.h>

#include "debug.h"
#include "cuda/helper_cuda.h"
#include "renderer.h"

Camera::Camera(Resolution const & _res, cv::Mat const & _mat)
  : __resolution(_res)
  , __frame(_mat)
{
  __flow = cv::Mat::zeros(__resolution.width, __resolution.height, CV_32FC2);
}

NullCamera::NullCamera(Resolution _res)
  : Camera(_res, cv::Mat::zeros(_res.height, _res.width, CV_8UC3))
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
  // for(int i = 0; i < 10; ++i) {
  //   for(int j = 0; j < 10; ++j) {
  //     {
  //       cv::Vec3b & e = frame().at<cv::Vec3b>(j, i);
  //       e[0] = 255u;
  //       e[1] = 255u;
  //       e[2] = 255u;
  //     }
  //     {
  //       cv::Vec3b & e = frame().at<cv::Vec3b>(_res.height - j, _res.width - i);
  //       e[0] = 255u;
  //       e[1] = 255u;
  //       e[2] = 255u;
  //     }
  //   }
  // }
  // for(int i = 0; i < 10; ++i) {
  //   for(int j = 0; j < 10; ++j) {
  //     cv::Vec3b & e = frame().at<cv::Vec3b>(j, i);
  //     e[0] = 255u;
  //     e[1] = 255u;
  //     e[2] = 255u;
  //   }
  // }
}

CVCamera::CVCamera(bool _optical_flow, int _index, Resolution _res, float _fps)
  : __camera(_index)
  , __capture(_optical_flow)
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
  frame().copyTo(lastFrame());
  flow() = cv::Mat::zeros(resolution().width, resolution().height, CV_32FC2);
  __capture.start(__camera, frame(), lastFrame(), flow(), time_delta());
}

void CVCamera::Thread::start(cv::VideoCapture & _cam, cv::Mat & _frame, cv::Mat & _last_frame, cv::Mat & _flow, float & _dt) {
  __continue = true;
  __last_ticks = SDL_GetTicks();
  __capture = std::thread(&Thread::capture, this, std::ref(_cam), std::ref(_frame), std::ref(_last_frame), std::ref(_flow), std::ref(_dt));
}

void CVCamera::Thread::stop() {
  if(__continue) {
    __continue = false;
    __capture.join();
  }
}

void CVCamera::Thread::capture(cv::VideoCapture & _cam, cv::Mat & _frame, cv::Mat & _last_frame, cv::Mat & _flow, float & _dt) {
  while(__continue) {
    if(_cam.grab()){
      _frame.copyTo(_last_frame);
      _cam.retrieve(_frame);
      auto ticks = SDL_GetTicks();
      _dt = (ticks - __last_ticks) / 1000.0f;
      __last_ticks = ticks;
      if(OPTICAL_FLOW) {
        cv::optflow::calcOpticalFlowSparseToDense(_last_frame, _frame, _flow);
      }
    }
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
