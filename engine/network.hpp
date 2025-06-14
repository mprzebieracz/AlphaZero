#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <torch/torch.h>
#include <vector>
#include <string>


struct ResidualBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

    ResidualBlockImpl(int channels) {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, /*kernel_size=*/3)
                .stride(1).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                .stride(1).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        x = torch::relu(bn1(conv1(x)));
        x = bn2(conv2(x));
        x += residual;
        return torch::relu(x);
    }
};
TORCH_MODULE(ResidualBlock);


struct AlphaZeroNetworkImpl : torch::nn::Module {
    int height, width;
    // stem
    torch::nn::Conv2d   conv_in{nullptr};
    torch::nn::BatchNorm2d bn_in{nullptr};
    torch::nn::Sequential residuals{nullptr};

    // policy head
    torch::nn::Conv2d   policy_conv{nullptr};
    torch::nn::BatchNorm2d policy_bn{nullptr};
    torch::nn::Linear   policy_fc{nullptr};

    // value head
    torch::nn::Conv2d   value_conv{nullptr};
    torch::nn::BatchNorm2d value_bn{nullptr};
    torch::nn::Linear   value_fc1{nullptr}, value_fc2{nullptr};

    AlphaZeroNetworkImpl(int input_channels,
                         int h,
                         int w,
                         int num_res_blocks,
                         int action_size,
                         int num_filters)
    : height(h), width(w)
    {
        conv_in = register_module("conv_in",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_filters, 3)
                .stride(1).padding(1).bias(false)));
        bn_in = register_module("bn_in", torch::nn::BatchNorm2d(num_filters));

        residuals = register_module("residuals", torch::nn::Sequential());
        for (int i = 0; i < num_res_blocks; ++i) {
            residuals->push_back(ResidualBlock(num_filters));
        }

        policy_conv = register_module("policy_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 2, 1).bias(false)));
        policy_bn   = register_module("policy_bn", torch::nn::BatchNorm2d(2));
        policy_fc   = register_module("policy_fc",
            torch::nn::Linear(2 * height * width, action_size));

        value_conv  = register_module("value_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 1, 1).bias(false)));
        value_bn    = register_module("value_bn", torch::nn::BatchNorm2d(1));
        value_fc1   = register_module("value_fc1",
            torch::nn::Linear(height * width, num_filters));
        value_fc2   = register_module("value_fc2",
            torch::nn::Linear(num_filters, 1));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(bn_in(conv_in(x)));
        x = residuals->forward(x);

        // policy head
        auto p = torch::relu(policy_bn(policy_conv(x)));
        p = p.view({p.size(0), -1});
        p = policy_fc(p);

        // value head
        auto v = torch::relu(value_bn(value_conv(x)));
        v = v.view({v.size(0), -1});
        v = torch::relu(value_fc1(v));
        v = torch::tanh(value_fc2(v));

        return {p, v};
    }
};
TORCH_MODULE(AlphaZeroNetwork);


#endif // NETWORK_HPP