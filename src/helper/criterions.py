import torch


def WGANGP_gradient_penalty(input, fake, discriminator, weight, backward=True):
    batchSize = input.size(0)
    alpha = torch.rand(batchSize, 1)
    alpha = alpha.expand(batchSize, int(input.nelement() /
                                        batchSize)).contiguous().view(
                                            input.size())
    alpha = alpha.to(input.device)
    interpolates = alpha * input + ((1 - alpha) * fake)

    interpolates = torch.autograd.Variable(
        interpolates, requires_grad=True)

    decisionInterpolate, labels = discriminator(interpolates)
    decisionInterpolate = decisionInterpolate[:, 0].sum()

    gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                    inputs=interpolates,
                                    create_graph=True, retain_graph=True)

    gradients = gradients[0].view(batchSize, -1)
    gradients = (gradients * gradients).sum(dim=1).sqrt()
    gradient_penalty = (((gradients - 1.0)**2)).sum() * weight

    if backward:
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()

def WGANGP_loss(discriminator_output, should_be_real: bool):
    discriminator_output_sum = discriminator_output[:, 0].sum()
    if should_be_real:
        return -discriminator_output_sum
    else:
        return discriminator_output_sum


def Epsilon_loss(prediction_real_data, epsilon_d):
    if epsilon_d > 0:
        return (prediction_real_data[:, 0] ** 2).sum() * epsilon_d
    else:
        return 0
