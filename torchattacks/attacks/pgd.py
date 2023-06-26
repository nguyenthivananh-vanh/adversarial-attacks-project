import torch
import torch.nn as nn

from ..attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, random_start=True):
        super().__init__("PGD", model) # ke thua tham so ten cuoc tan cong va model tu lop cha
        # bo sung them cac tham so khac
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
       
        loss = nn.CrossEntropyLoss()
        # tach va luu tru vi du doi khang
        adv_images = images.clone().detach()

        if self.random_start:
            # cong them ma tran nhieu co cung kich thuoc trong khoang [-eps,eps]
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            # ep gia tri vao khoang [0,1] 
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True # cho phep tinh toan gradient
            # lay ra cac diem so cua mo hinh voi dau vao la adv_images
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            # di chuyen mot khoang alpha theo huong tang truong nhanh nhat cua cost
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            # tinh su khac biet giua anh bi nhieu va anh ban dau, ép vao khoang [-eps, eps]
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # cong voi gia tri delta va ep vao khoang [0,1]
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
