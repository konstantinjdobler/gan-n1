import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

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

def WGANGP_loss(prediction_fake_data, prediction_real_data=None):
        fake_data_sum = prediction_fake_data[:, 0].sum()
        if prediction_real_data is not None:
            real_data_sum = prediction_real_data[:, 0].sum()
            return fake_data_sum - real_data_sum
        else:
            return -fake_data_sum

def Epsilon_loss(prediction_real_data, epsilon_d):
    if epsilon_d > 0:
        return (prediction_real_data[:, 0] ** 2).sum() * epsilon_d
    else:
        return 0


def MultiLabelClassificationLoss(predicted_labels, real_labels):
    loss = torch.nn.BCELoss()
    real_labels[real_labels == -1] = 0
    return loss(predicted_labels, real_labels)

class ACGANCriterion:
    @staticmethod
    def set_key_order(attrib_keys_order, device):
        ACGANCriterion.device = device
        ACGANCriterion.number_of_attributes = len(attrib_keys_order)
        ACGANCriterion.attributes_size = [0 for i in range(ACGANCriterion.number_of_attributes)]
        ACGANCriterion.key_order = ['' for x in range(ACGANCriterion.number_of_attributes)]
        ACGANCriterion.labels_order = {}

        ACGANCriterion.input_dict = deepcopy(attrib_keys_order)

        for key in attrib_keys_order:
            order = attrib_keys_order[key]["order"]
            ACGANCriterion.key_order[order] = key
            ACGANCriterion.attributes_size[order] = len(attrib_keys_order[key]["values"])
            ACGANCriterion.labels_order[key] = {index: label for label, index in
                                     enumerate(attrib_keys_order[key]["values"])}

        ACGANCriterion.label_weights = torch.tensor(
            [1.0 for x in range(ACGANCriterion.get_input_dim())])

        for key in attrib_keys_order:
            order = attrib_keys_order[key]["order"]
            if attrib_keys_order[key].get('weights', None) is not None:
                shift = sum(ACGANCriterion.attributes_size[:order])

                for value, weight in attrib_keys_order[key]['weights'].items():
                    ACGANCriterion.label_weights[shift + ACGANCriterion.labels_order[key][value]] = weight

        ACGANCriterion.size_output = ACGANCriterion.number_of_attributes

    @staticmethod
    def get_input_dim():
        return sum(ACGANCriterion.attributes_size)

    @staticmethod
    def loss(output_discriminator, target_labels):
        loss = 0
        shift_input = 0
        shift_target = 0

        label_weights = ACGANCriterion.label_weights.to(ACGANCriterion.device)

        for i in range(ACGANCriterion.number_of_attributes):
            C = ACGANCriterion.attributes_size[i]
            loc_input = output_discriminator[:, shift_input:(shift_input+C)]
            loc_target = target_labels[:, shift_target].long()
            #loc_loss = F.cross_entropy(loc_input, loc_target, weight=label_weights[shift_input:(shift_input+C)])
            loc_loss = F.cross_entropy(loc_input, loc_target)
            shift_target += 1
            loss += loc_loss
            shift_input += C

        return loss

    @staticmethod
    def build_random_criterion_tensor(size_batch):
        target_out = []
        input_latent = []

        # for i in range(ACGANCriterion.number_of_attributes):
        #     C = ACGANCriterion.attributes_size[i]
        #     v = np.random.randint(0, C, size_batch)
        #     w = np.zeros((size_batch, C), dtype='float32')
        #     w[np.arange(size_batch), v] = 1
        #     y = torch.tensor(w).view(size_batch, C)

        #     input_latent.append(y)
        #     target_out.append(torch.tensor(v).float().view(size_batch, 1))

        # for _ in range(size_batch):
        #     random_labels = [random.choice([-1, 1]) for __ in range(ACGANCriterion.number_of_attributes)]
        #     y = torch.tensor(random_labels).view(size_batch)
        #     input_latent.append(y)
        
        for i in range(ACGANCriterion.number_of_attributes):
            C = ACGANCriterion.attributes_size[i]
            v = np.random.randint(0, C, size_batch)
            w = np.zeros((size_batch, C), dtype='float32')
            w[np.arange(size_batch), v] = 1
            y = torch.tensor(w).view(size_batch, C)

            input_latent.append(y)
            target_out.append(torch.tensor(v).float().view(size_batch, 1))

        return torch.cat(target_out, dim=1), torch.cat(input_latent, dim=1)

    @staticmethod
    def build_latent_criterion(target_cat):

        batch_size = target_cat.size(0)
        idx = torch.arange(batch_size, device=ACGANCriterion.device)
        target_out = torch.zeros((batch_size, sum(ACGANCriterion.attributes_size)))
        shift = 0

        for i in range(ACGANCriterion.number_of_attributes):
            target_out[idx, shift + target_cat[:, i]] = 1
            shift += ACGANCriterion.attributes_size[i]

        return target_out