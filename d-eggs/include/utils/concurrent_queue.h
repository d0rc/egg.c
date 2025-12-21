#ifndef EGG_CONCURRENT_QUEUE_H
#define EGG_CONCURRENT_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename T>
class ConcurrentQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    bool cancelled_ = false;

public:
    ConcurrentQueue() = default;
    ConcurrentQueue(const ConcurrentQueue&) = delete;
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cond_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || cancelled_; });
        if (queue_.empty() && cancelled_) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void cancel() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cancelled_ = true;
        }
        cond_.notify_all();
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

#endif // EGG_CONCURRENT_QUEUE_H
