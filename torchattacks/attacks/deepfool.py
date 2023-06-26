import torch
import torch.nn as nn

from ..attack import Attack


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, steps=50, overshoot=0.02):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self.supported_mode = ['default']

    def forward(self, images, labels, return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        #luu tru so luong anh
        batch_size = len(images)
        #luu tru trang thai cua tung anh da duoc tan cong chua
        correct = torch.tensor([True]*batch_size)
        #luu tru nhan muc tieu cua tung anh trong batch
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:
            return adv_images, target_labels

        return adv_images

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.get_logits(image)[0] #lay mang gia tri logit cua cac nhan
        _, pre = torch.max(fs, dim=0) #tim gia tri va chi so lon nhat trong fs theo chieu thu 0 va luu vao pre (nhan du doan)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image) #lưu trữ các vector gradient cho từng nhãn trong fs.
        image = image.detach()

        f_0 = fs[label] #lay gia tri logit cua nhan dung
        w_0 = ws[label] # lay vector gradient tuong ung voi nhan dung
        # lưu trữ các chỉ số của các nhãn sai
        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes] #lưu trữ các giá trị logit cho các nhãn sai
        w_k = ws[wrong_classes] #lưu trữ các vector gradient cho các nhãn sai.

        f_prime = f_k - f_0 #lưu trữ hiệu của các giá trị logit.
        w_prime = w_k - w_0 # lưu trữ hiệu của các vector gradient.
        #lưu trữ tỷ số giữa hiệu logit và hiệu gradient.
        value = torch.abs(f_prime) \
                / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0) #tìm ra giá trị và chỉ số nhỏ nhất trong tensor value,
        #lưu trữ chỉ số của nhãn mục tiêu trong danh sách wrong_classes.
        #tính toán nhiễu đối kháng 
        delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2)**2))
        #tính toán nhãn mục tiêu theo chỉ số hat_L và gán cho biến target_label
        target_label = hat_L if hat_L < label else hat_L+1
        #cap nhat nhieu
        adv_image = image + (1+self.overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx+1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
