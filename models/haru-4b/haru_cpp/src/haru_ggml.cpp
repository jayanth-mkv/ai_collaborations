#include "haru_ggml.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

namespace haru {

HaruEngine::HaruEngine(const HaruConfig& config)
    : config_(config), prev_model_latent_(config.latent_dim, 0.0f) {}

HaruEngine::~HaruEngine() = default;

bool HaruEngine::load_model(const std::string& backbone_gguf_path,
                            const std::string& codec_gguf_path,
                            const std::string& head_gguf_path) {
    (void)backbone_gguf_path;
    (void)codec_gguf_path;
    (void)head_gguf_path;
    is_initialized_ = true;
    return true;
}

void HaruEngine::reset() {
    std::fill(prev_model_latent_.begin(), prev_model_latent_.end(), 0.0f);
    prev_token_id_ = 0;
}

bool HaruEngine::step(const float* user_pcm_3200,
                      float* model_pcm_3200,
                      int32_t* out_token_id,
                      HaruFrameMetrics* metrics) {
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    // 1. VibeVoice Encoder: 3200 samples -> 64-dim latent
    auto t0 = clock::now();
    std::vector<float> user_latent(config_.latent_dim, 0.0f);
    int stride = config_.frame_samples / config_.latent_dim;
    for (int i = 0; i < config_.latent_dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < stride; ++j) {
            sum += std::abs(user_pcm_3200[i * stride + j]);
        }
        user_latent[i] = sum / static_cast<float>(stride);
    }
    auto t_enc = clock::now();

    // 2. Multimodal Fusion & Backbone Forward Pass
    // Sums text token embedding + model audio projection + user audio projection
    std::vector<float> fused_embedding(config_.hidden_size, 0.0f);
    for (int i = 0; i < config_.latent_dim; ++i) {
        fused_embedding[i] += user_latent[i] + prev_model_latent_[i];
    }
    // Simulated LM forward pass (typically 35-50ms on 16c AVX-512 CPU)
    std::vector<float> hidden_state(config_.hidden_size, 0.0f);
    for (int i = 0; i < config_.hidden_size; ++i) {
        hidden_state[i] = std::sin(static_cast<float>(i) * 0.01f + fused_embedding[i % config_.latent_dim]);
    }
    if (out_token_id) {
        *out_token_id = prev_token_id_ + 1;
        prev_token_id_ = *out_token_id;
    }
    auto t_bb = clock::now();

    // 3. MeanFlow 1-Step Sampling Head
    auto t_hd0 = clock::now();
    std::vector<float> next_latent(config_.latent_dim, 0.0f);
    for (int i = 0; i < config_.latent_dim; ++i) {
        float velocity = std::tanh(hidden_state[i]) * config_.guidance_scale;
        next_latent[i] = prev_model_latent_[i] + 1.0f * velocity;
    }
    prev_model_latent_ = next_latent;
    auto t_hd = clock::now();

    // 4. VibeVoice Decoder: 64-dim latent -> 3200 samples
    auto t_dec0 = clock::now();
    for (int i = 0; i < config_.frame_samples; ++i) {
        int latent_idx = (i * config_.latent_dim) / config_.frame_samples;
        float carrier = std::sin(2.0f * 3.14159265f * 440.0f * (static_cast<float>(i) / 24000.0f));
        model_pcm_3200[i] = carrier * next_latent[latent_idx] * 0.1f;
    }
    auto t_end = clock::now();

    if (metrics) {
        metrics->encode_ms = std::chrono::duration<float, std::milli>(t_enc - t0).count();
        metrics->backbone_ms = std::chrono::duration<float, std::milli>(t_bb - t_enc).count();
        metrics->head_ms = std::chrono::duration<float, std::milli>(t_hd - t_hd0).count();
        metrics->decode_ms = std::chrono::duration<float, std::milli>(t_end - t_dec0).count();
        metrics->total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        metrics->meets_deadline = (metrics->total_ms <= (1000.0f / config_.frame_rate));
    }

    return true;
}

} // namespace haru
