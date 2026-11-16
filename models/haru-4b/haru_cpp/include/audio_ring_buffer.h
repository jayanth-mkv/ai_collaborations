#pragma once

#include <vector>
#include <mutex>
#include <algorithm>
#include <cstddef>

namespace haru {

/**
 * Thread-safe ring buffer for streaming 24 kHz audio chunks (e.g. 3200 samples per 133ms frame).
 */
class AudioRingBuffer {
public:
    explicit AudioRingBuffer(size_t capacity_samples = 3200 * 50)
        : buffer_(capacity_samples, 0.0f), capacity_(capacity_samples), head_(0), tail_(0), size_(0) {}

    void write(const float* data, size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < count; ++i) {
            buffer_[head_] = data[i];
            head_ = (head_ + 1) % capacity_;
            if (size_ < capacity_) {
                size_++;
            } else {
                tail_ = (tail_ + 1) % capacity_; // Overwrite oldest on overflow
            }
        }
    }

    bool read_frame(float* out_data, size_t frame_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ < frame_size) {
            return false;
        }
        for (size_t i = 0; i < frame_size; ++i) {
            out_data[i] = buffer_[tail_];
            tail_ = (tail_ + 1) % capacity_;
        }
        size_ -= frame_size;
        return true;
    }

    size_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        head_ = 0;
        tail_ = 0;
        size_ = 0;
    }

private:
    std::vector<float> buffer_;
    size_t capacity_;
    size_t head_;
    size_t tail_;
    size_t size_;
    mutable std::mutex mutex_;
};

} // namespace haru
