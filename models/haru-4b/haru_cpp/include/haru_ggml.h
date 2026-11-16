#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace haru {

struct HaruConfig {
    int sample_rate = 24000;
    int frame_samples = 3200;
    float frame_rate = 7.5f;
    int latent_dim = 64;
    int hidden_size = 2560;
    int vocab_size = 248320;
    int num_layers = 32;
    float guidance_scale = 1.5f;
};

struct HaruFrameMetrics {
    float encode_ms = 0.0f;
    float backbone_ms = 0.0f;
    float head_ms = 0.0f;
    float decode_ms = 0.0f;
    float total_ms = 0.0f;
    bool meets_deadline = true; // <= 133.33 ms
};

class HaruEngine {
public:
    explicit HaruEngine(const HaruConfig& config = HaruConfig{});
    ~HaruEngine();

    bool load_model(const std::string& backbone_gguf_path,
                    const std::string& codec_gguf_path,
                    const std::string& head_gguf_path);

    /**
     * Executes one 133.33 ms duplex frame:
     * - Encodes 3200 samples of user audio into a 64-dim latent.
     * - Fuses user latent, model latent, and text token.
     * - Runs LM backbone forward pass.
     * - Generates next latent using MeanFlow head.
     * - Decodes next latent into 3200 samples of model speech.
     */
    bool step(const float* user_pcm_3200,
              float* model_pcm_3200,
              int32_t* out_token_id,
              HaruFrameMetrics* metrics = nullptr);

    void reset();

    const HaruConfig& config() const { return config_; }

private:
    HaruConfig config_;
    std::vector<float> prev_model_latent_;
    int32_t prev_token_id_ = 0;
    bool is_initialized_ = false;
};

} // namespace haru
