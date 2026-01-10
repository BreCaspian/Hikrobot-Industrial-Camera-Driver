#include "hik_camera/hik_camera.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>

#ifdef HIK_CAMERA_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef HIK_CAMERA_WITH_OPENCV_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif
#ifdef HIK_CAMERA_WITH_NPP
#include "hik_camera/gpu_demosaic_npp.h"
#endif

namespace hik_camera {
namespace {
std::atomic<bool> g_pinned_pool_logged{false};
std::atomic<bool> g_gpu_pipeline_warned{false};
std::atomic<bool> g_gpu_demosaic_warned{false};
std::atomic<bool> g_gpu_demosaic_failed{false};
std::atomic<bool> g_gpu_demosaic_logged{false};
std::atomic<bool> g_gpu_demosaic_backend_warned{false};
std::atomic<bool> g_cuda_device_warned{false};
std::atomic<bool> g_cuda_memcpy_h2d_logged{false};
std::atomic<bool> g_cuda_memcpy_d2h_logged{false};
std::atomic<bool> g_gpu_post_warned{false};
std::atomic<bool> g_gpu_post_logged{false};
std::atomic<uint64_t> g_gpu_post_last_ms{0};
std::atomic<uint64_t> g_gpu_post_frames{0};

bool is_bayer8(uint32_t pixel_type) {
    return pixel_type == PixelType_Gvsp_BayerGR8 ||
        pixel_type == PixelType_Gvsp_BayerRG8 ||
        pixel_type == PixelType_Gvsp_BayerGB8 ||
        pixel_type == PixelType_Gvsp_BayerBG8 ||
        pixel_type == PixelType_Gvsp_HB_BayerGR8 ||
        pixel_type == PixelType_Gvsp_HB_BayerRG8 ||
        pixel_type == PixelType_Gvsp_HB_BayerGB8 ||
        pixel_type == PixelType_Gvsp_HB_BayerBG8;
}

#if defined(HIK_CAMERA_WITH_OPENCV_CUDA)
int bayer_code_from_pattern(const std::string& pattern) {
    std::string p = pattern;
    std::transform(p.begin(), p.end(), p.begin(),
        [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (p == "BG") {
        return cv::COLOR_BayerBG2BGR;
    }
    if (p == "GB") {
        return cv::COLOR_BayerGB2BGR;
    }
    if (p == "GR") {
        return cv::COLOR_BayerGR2BGR;
    }
    if (p == "RG") {
        return cv::COLOR_BayerRG2BGR;
    }
    return cv::COLOR_BayerGB2BGR;
}

int bayer_code_from_pixel_type(uint32_t pixel_type) {
    switch (pixel_type) {
        case PixelType_Gvsp_BayerBG8:
        case PixelType_Gvsp_HB_BayerBG8:
            return cv::COLOR_BayerBG2BGR;
        case PixelType_Gvsp_BayerGB8:
        case PixelType_Gvsp_HB_BayerGB8:
            return cv::COLOR_BayerGB2BGR;
        case PixelType_Gvsp_BayerGR8:
        case PixelType_Gvsp_HB_BayerGR8:
            return cv::COLOR_BayerGR2BGR;
        case PixelType_Gvsp_BayerRG8:
        case PixelType_Gvsp_HB_BayerRG8:
            return cv::COLOR_BayerRG2BGR;
        default:
            return cv::COLOR_BayerGB2BGR;
    }
}
#endif
}  // namespace

bool HikCamera::applyGpuPostProcess(Frame& frame,
    cv::cuda::GpuMat& bgr_in,
    cv::cuda::Stream& stream,
    uint8_t* bgr_dst,
    size_t bgr_bytes) {
#if defined(HIK_CAMERA_WITH_OPENCV_CUDA)
    if (!(config_.gpu_undistort_enable || config_.gpu_rotate_180 || config_.gpu_resize_enable)) {
        return false;
    }
    cv::cuda::GpuMat* current = &bgr_in;

    if (config_.gpu_undistort_enable && !gpu_map1_.empty() && !gpu_map2_.empty()) {
        cv::cuda::remap(*current, gpu_post_, gpu_map1_, gpu_map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), stream);
        current = &gpu_post_;
    }

    if (config_.gpu_rotate_180) {
        cv::cuda::flip(*current, gpu_post_, -1, stream);
        current = &gpu_post_;
    }

    if (config_.gpu_resize_enable && config_.gpu_resize_width > 0 && config_.gpu_resize_height > 0) {
        if (!config_.gpu_output_only && !g_gpu_post_warned.exchange(true)) {
            std::cout << "GPU resize requires gpu_output_only=1; skip resize\n";
        } else if (config_.gpu_output_only) {
            cv::cuda::resize(*current, gpu_post_,
                cv::Size(config_.gpu_resize_width, config_.gpu_resize_height), 0.0, 0.0, cv::INTER_LINEAR, stream);
            current = &gpu_post_;
        }
    }

    frame.gpu_bgr_ptr = current->ptr<uint8_t>();
    frame.gpu_stride_bytes = static_cast<int>(current->step);
    frame.gpu_stream = cv::cuda::StreamAccessor::getStream(stream);
    frame.gpu_valid = true;

    if (!config_.gpu_output_only && bgr_dst && bgr_bytes > 0) {
        current->download(frame.bgr, stream);
        stream.waitForCompletion();
    }
    if (!g_gpu_post_logged.exchange(true)) {
        std::cout << "GPU post-process active (undistort/rotate/resize)\n";
    }
    uint64_t now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    uint64_t last_ms = g_gpu_post_last_ms.load();
    uint64_t frames = g_gpu_post_frames.fetch_add(1) + 1;
    if (last_ms == 0) {
        g_gpu_post_last_ms.store(now_ms);
        g_gpu_post_frames.store(0);
    } else if (now_ms - last_ms >= 1000) {
        if (g_gpu_post_last_ms.compare_exchange_strong(last_ms, now_ms)) {
            double fps = static_cast<double>(frames) * 1000.0 / static_cast<double>(now_ms - last_ms);
            g_gpu_post_frames.store(0);
            std::cout << "GPU post-process FPS: " << fps << "\n";
        }
    }
    return true;
#else
    (void)frame;
    (void)bgr_in;
    (void)stream;
    (void)bgr_dst;
    (void)bgr_bytes;
    if (!g_gpu_post_warned.exchange(true)) {
        std::cout << "GPU post-process requested but OpenCV CUDA not enabled; fallback to CPU\n";
    }
    return false;
#endif
}

void HikCamera::releaseGpuResources() {
    unregisterPinnedPool();
#ifdef HIK_CAMERA_WITH_CUDA
    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
    if (npp_raw_dev_) {
        cudaFree(npp_raw_dev_);
        npp_raw_dev_ = nullptr;
        npp_raw_bytes_ = 0;
    }
    if (npp_bgr_dev_) {
        cudaFree(npp_bgr_dev_);
        npp_bgr_dev_ = nullptr;
        npp_bgr_bytes_ = 0;
    }
#endif
}

void HikCamera::configureGpuPipeline() {
#ifdef HIK_CAMERA_WITH_CUDA
    auto cleanup_cuda_stream = [&]() {
        if (cuda_stream_) {
            cudaStreamDestroy(cuda_stream_);
            cuda_stream_ = nullptr;
        }
    };
    auto cleanup_npp_buffers = [&]() {
        if (npp_raw_dev_) {
            cudaFree(npp_raw_dev_);
            npp_raw_dev_ = nullptr;
            npp_raw_bytes_ = 0;
        }
        if (npp_bgr_dev_) {
            cudaFree(npp_bgr_dev_);
            npp_bgr_dev_ = nullptr;
            npp_bgr_bytes_ = 0;
        }
    };

    if (config_.gpu_pipeline_enable || config_.gpu_demosaic_enable) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
            int dev = std::max(0, std::min(config_.gpu_device_id, device_count - 1));
            if (cudaSetDevice(dev) != cudaSuccess && !g_cuda_device_warned.exchange(true)) {
                std::cerr << "Failed to set CUDA device " << dev << "\n";
            }
            if (config_.gpu_stream_enable) {
                if (!cuda_stream_) {
                    cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
                }
            } else {
                cleanup_cuda_stream();
            }
        } else if (!g_cuda_device_warned.exchange(true)) {
            std::cout << "CUDA device not available; GPU features disabled\n";
        }
    } else {
        cleanup_cuda_stream();
        cleanup_npp_buffers();
    }
#else
    if (config_.gpu_pipeline_enable && !g_gpu_pipeline_warned.exchange(true)) {
        std::cout << "GPU pipeline requested but CUDA not enabled; skip pinned pool\n";
    }
#endif
}

void HikCamera::unregisterPinnedPool() {
#ifdef HIK_CAMERA_WITH_CUDA
    if (bgr_pool_pinned_) {
        for (auto& buf : bgr_pool_) {
            if (!buf.empty()) {
                cudaHostUnregister(buf.data());
            }
        }
        bgr_pool_pinned_ = false;
    }
#endif
}

void HikCamera::configurePinnedPool() {
#ifdef HIK_CAMERA_WITH_CUDA
    if (config_.gpu_pipeline_enable && !bgr_pool_pinned_) {
        bool ok = true;
        for (auto& buf : bgr_pool_) {
            if (!buf.empty()) {
                auto ret = cudaHostRegister(buf.data(), buf.size(), cudaHostRegisterPortable);
                if (ret != cudaSuccess) {
                    ok = false;
                    break;
                }
            }
        }
        if (ok) {
            bgr_pool_pinned_ = true;
            if (!g_pinned_pool_logged.exchange(true)) {
                std::cout << "Pinned pool enabled (cudaHostRegister)\n";
            }
        } else {
            unregisterPinnedPool();
            std::cerr << "Pinned pool registration failed, fallback to pageable memory\n";
        }
    }
#else
    if (config_.gpu_pipeline_enable && !g_gpu_pipeline_warned.exchange(true)) {
        std::cout << "GPU pipeline requested but CUDA not enabled; skip pinned pool\n";
    }
#endif
}

bool HikCamera::tryGpuDemosaic(const MV_FRAME_OUT& out_frame,
    const MV_FRAME_OUT_INFO_EX& info,
    Frame& frame,
    uint8_t* bgr_dst) {
    (void)out_frame;
    (void)frame;
    (void)bgr_dst;
    if (!is_bayer8(info.enPixelType)) {
        if (!g_gpu_demosaic_warned.exchange(true)) {
            std::cout << "GPU demosaic requested but pixel type is not Bayer8; fallback to CPU\n";
        }
        return false;
    }

    std::string backend = config_.gpu_demosaic_backend;
    std::transform(backend.begin(), backend.end(), backend.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (backend.empty()) {
        backend = "auto";
    }
    if (backend != "auto" && backend != "opencv" && backend != "npp") {
        if (!g_gpu_demosaic_backend_warned.exchange(true)) {
            std::cout << "Unknown gpu_demosaic_backend='" << backend
                      << "', fallback to auto\n";
        }
        backend = "auto";
    }
    bool want_opencv = backend == "auto" || backend == "opencv";
    bool want_npp = backend == "auto" || backend == "npp";

    std::string pattern = config_.bayer_pattern;
    std::transform(pattern.begin(), pattern.end(), pattern.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (pattern == "auto") {
        pattern.clear();
    }

#if defined(HIK_CAMERA_WITH_OPENCV_CUDA)
    if (want_opencv) {
        if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
            if (backend == "opencv" && !g_gpu_demosaic_warned.exchange(true)) {
                std::cout << "GPU demosaic requested but no CUDA device; fallback to CPU\n";
            }
        } else {
            try {
                int code = pattern.empty()
                    ? bayer_code_from_pixel_type(info.enPixelType)
                    : bayer_code_from_pattern(pattern);
                cv::Mat raw(frame.height, frame.width, CV_8UC1, out_frame.pBufAddr);
                cv::cuda::Stream stream = config_.gpu_stream_enable ? gpu_stream_ : cv::cuda::Stream();
                gpu_raw_.upload(raw, stream);
                cv::cuda::demosaicing(gpu_raw_, gpu_bgr_, code, -1, stream);
                bool post_done = applyGpuPostProcess(frame, gpu_bgr_, stream, bgr_dst,
                    static_cast<size_t>(frame.width) * frame.height * 3);
                if (!post_done) {
                    frame.gpu_bgr_ptr = gpu_bgr_.ptr<uint8_t>();
                    frame.gpu_stride_bytes = static_cast<int>(gpu_bgr_.step);
                    frame.gpu_stream = cv::cuda::StreamAccessor::getStream(stream);
                    frame.gpu_valid = true;
                    if (!config_.gpu_output_only) {
                        gpu_bgr_.download(frame.bgr, stream);
                        stream.waitForCompletion();
                    }
                }
                if (!g_gpu_demosaic_logged.exchange(true)) {
                    std::cout << "GPU demosaic active (opencv)\n";
                }
                return true;
            } catch (const cv::Exception& e) {
                if (backend == "opencv" && !g_gpu_demosaic_failed.exchange(true)) {
                    std::cerr << "GPU demosaic failed; fallback to CPU: " << e.what() << "\n";
                }
            }
        }
    }
#else
    if (want_opencv && backend == "opencv" && !g_gpu_demosaic_warned.exchange(true)) {
        std::cout << "GPU demosaic requested but OpenCV CUDA not enabled; fallback to CPU\n";
    }
#endif

#if defined(HIK_CAMERA_WITH_NPP)
    if (want_npp) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
            if (backend == "npp" && !g_gpu_demosaic_warned.exchange(true)) {
                std::cout << "GPU demosaic requested but no CUDA device; fallback to CPU\n";
            }
        } else {
            size_t raw_bytes = static_cast<size_t>(frame.width) * frame.height;
            size_t bgr_bytes = static_cast<size_t>(frame.width) * frame.height * 3;
            if (npp_raw_bytes_ != raw_bytes) {
                if (npp_raw_dev_) {
                    cudaFree(npp_raw_dev_);
                    npp_raw_dev_ = nullptr;
                }
                if (cudaMalloc(&npp_raw_dev_, raw_bytes) != cudaSuccess) {
                    npp_raw_dev_ = nullptr;
                }
                npp_raw_bytes_ = npp_raw_dev_ ? raw_bytes : 0;
            }
            if (npp_bgr_bytes_ != bgr_bytes) {
                if (npp_bgr_dev_) {
                    cudaFree(npp_bgr_dev_);
                    npp_bgr_dev_ = nullptr;
                }
                if (cudaMalloc(&npp_bgr_dev_, bgr_bytes) != cudaSuccess) {
                    npp_bgr_dev_ = nullptr;
                }
                npp_bgr_bytes_ = npp_bgr_dev_ ? bgr_bytes : 0;
            }

            if (!npp_raw_dev_ || !npp_bgr_dev_) {
                if (backend == "npp" && !g_gpu_demosaic_failed.exchange(true)) {
                    std::cerr << "GPU demosaic failed; CUDA allocation error\n";
                }
            } else {
                cudaStream_t stream = (config_.gpu_stream_enable && cuda_stream_) ? cuda_stream_ : nullptr;
                cudaError_t h2d_status = cudaMemcpyAsync(
                    npp_raw_dev_, out_frame.pBufAddr, raw_bytes, cudaMemcpyHostToDevice, stream);
                if (h2d_status == cudaSuccess && !g_cuda_memcpy_h2d_logged.exchange(true)) {
                    std::cout << "cudaMemcpyAsync H2D ok (npp_raw_dev_)\n";
                }
                int npp_code = pattern.empty()
                    ? nppBayerFromPixelType(info.enPixelType)
                    : nppBayerFromPattern(pattern);
                std::string err;
                if (nppDemosaicBayer8(npp_raw_dev_,
                        static_cast<int>(frame.width),
                        npp_bgr_dev_,
                        static_cast<int>(frame.width) * 3,
                        static_cast<int>(frame.width),
                        static_cast<int>(frame.height),
                        npp_code,
                        stream,
                        &err)) {
                    frame.gpu_bgr_ptr = npp_bgr_dev_;
                    frame.gpu_stride_bytes = static_cast<int>(frame.width) * 3;
                    frame.gpu_stream = stream;
                    frame.gpu_valid = true;
#if defined(HIK_CAMERA_WITH_OPENCV_CUDA)
                    if (config_.gpu_undistort_enable || config_.gpu_rotate_180 || config_.gpu_resize_enable) {
                        cv::cuda::Stream cv_stream = (stream != nullptr)
                            ? cv::cuda::StreamAccessor::wrapStream(stream)
                            : (config_.gpu_stream_enable ? gpu_stream_ : cv::cuda::Stream());
                        gpu_bgr_.create(frame.height, frame.width, CV_8UC3);
                        cudaMemcpyAsync(gpu_bgr_.ptr<uint8_t>(), npp_bgr_dev_, bgr_bytes,
                            cudaMemcpyDeviceToDevice, stream);
                        bool post_done = applyGpuPostProcess(frame, gpu_bgr_, cv_stream, bgr_dst, bgr_bytes);
                        if (post_done) {
                            if (!g_gpu_demosaic_logged.exchange(true)) {
                                std::cout << "GPU demosaic active (npp)\n";
                            }
                            return true;
                        }
                    }
#endif
                    if (!config_.gpu_output_only) {
                        cudaError_t d2h_status = cudaMemcpyAsync(
                            bgr_dst, npp_bgr_dev_, bgr_bytes, cudaMemcpyDeviceToHost, stream);
                        if (d2h_status == cudaSuccess && !g_cuda_memcpy_d2h_logged.exchange(true)) {
                            std::cout << "cudaMemcpyAsync D2H ok (npp_bgr_dev_)\n";
                        }
                        cudaStreamSynchronize(stream);
                    }
                    if (!g_gpu_demosaic_logged.exchange(true)) {
                        std::cout << "GPU demosaic active (npp)\n";
                    }
                    return true;
                }
                if (backend == "npp" && !g_gpu_demosaic_failed.exchange(true)) {
                    std::cerr << "GPU demosaic failed; fallback to CPU: " << err << "\n";
                }
            }
        }
    }
#else
    if (want_npp && backend == "npp" && !g_gpu_demosaic_warned.exchange(true)) {
        std::cout << "GPU demosaic requested but NPP not enabled; fallback to CPU\n";
    }
#endif

    if (backend == "auto" && !g_gpu_demosaic_warned.exchange(true)) {
        std::cout << "GPU demosaic requested but no backend available; fallback to CPU\n";
    }
    return false;
}

}  // namespace hik_camera
