#include "haru_ggml.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "================================================================================" << std::endl;
    std::cout << "  HARU-4B C++ RUNTIME HARNESS (7.5 Hz / 133.33 ms frame budget)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    haru::HaruEngine engine;
    const int total_frames = 20;

    std::vector<float> user_mic(3200, 0.05f);
    std::vector<float> model_speaker(3200, 0.0f);
    int32_t token_id = 0;

    std::cout << std::left 
              << std::setw(8) << "Frame"
              << std::setw(14) << "Encode(ms)"
              << std::setw(16) << "Backbone(ms)"
              << std::setw(12) << "Head(ms)"
              << std::setw(14) << "Decode(ms)"
              << std::setw(14) << "Total(ms)"
              << "Status" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::vector<float> totals;
    int met_count = 0;

    for (int i = 1; i <= total_frames; ++i) {
        haru::HaruFrameMetrics m;
        engine.step(user_mic.data(), model_speaker.data(), &token_id, &m);
        totals.push_back(m.total_ms);
        if (m.meets_deadline) met_count++;

        std::cout << std::left
                  << std::setw(8) << i
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << m.encode_ms
                  << std::setw(16) << m.backbone_ms
                  << std::setw(12) << m.head_ms
                  << std::setw(14) << m.decode_ms
                  << std::setw(14) << m.total_ms
                  << (m.meets_deadline ? "[OK]" : "[DEADLINE MISSED]") << std::endl;
    }

    std::cout << std::string(80, '-') << std::endl;
    float avg = std::accumulate(totals.begin(), totals.end(), 0.0f) / totals.size();
    std::cout << "Summary: Frames: " << total_frames 
              << " | Avg Latency: " << avg << " ms"
              << " | Target Deadline: 133.33 ms"
              << " | Real-Time Factor (RTF): " << (avg / 133.33f)
              << " | Real-Time Success Rate: " << (met_count * 100.0f / total_frames) << "%"
              << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
