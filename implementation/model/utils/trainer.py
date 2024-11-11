import warnings
import torch
from tqdm import tqdm, trange
from fgpgd import functions
import numpy as np
from PIL import Image
import os


mid_output = None

class Trainer:
    """
    Trainer class that eases the training of a PyTorch model.
    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    criterion : torch.nn.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    epochs : int
        The total number of iterations of all the training
        data in one cycle for training the model.
    scaler : torch.cuda.amp
        The parameter can be used to normalize PyTorch Tensors
        using native functions more detail:
        https://pytorch.org/docs/stable/index.html.
    lr_scheduler : torch.optim.lr_scheduler
        A predefined framework that adjusts the learning rate
        between epochs or iterations as the training progresses.
    Attributes
    ----------
    train_losses_ : torch.tensor
        It is a log of train losses for each epoch step.
    val_losses_ : torch.tensor
        It is a log of validation losses for each epoch step.
    """

    def __init__(
            self,
            model,
            criterion,
            epochs,
            metrics=None,
            initial_metrics=None,
            actual_metrics=None,
            logger=None,
            model_save_path=None,
            args=None,
            scaler=None,
            lr_scheduler=None,
            device=None,
            num_classes=None

    ):
        self.criterion = criterion
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.device = self._get_device(device)
        self.epochs = epochs
        self.logger = logger
        self.model = model.to(self.device)
        self.metrics = metrics

        self.model_save_path = model_save_path
        self.mIoU = 0.0
        self.mode = args.mode
        self.epsilon = args.epsilon
        # self.alpha = -1*args.alpha if self.targeted else args.alpha
        self.alpha = args.alpha
        self.iterations = args.iterations
        self.attack = args.attack
        self.num_classes = num_classes
        self.norm = args.norm
        self.targeted = args.targeted
        self.batch_size = None
        self.initial_metrics = initial_metrics
        self.actual_metrics = actual_metrics
        self.cosine = args.cosine
        self.data_name = args.dataset
        self.results_path = args.results_path
        if self.mode == 'adv_attack':
            self.source_layer = args.source_layer
        else:
            self.source_layer = None
        self.attack_imgs = []
        self.lamda = args.lamda
        if self.mode == 'trans_test':
            self.target_model = args.target_model
        else:
            self.target_model = args.source_model

    def save_ckpt(self, epoch):
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": self.train_losses_,
                    "val_loss": self.val_losses_,
                    "mIoU": self.mIoU},
                   self.model_save_path)

    # FGSM attack code
    def fgsm_attack(self, perturbed_image, data_grad, orig_image):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        if self.targeted:
            sign_data_grad *= -1
        perturbed_image = perturbed_image.detach() + self.alpha * sign_data_grad
        # Adding clipping to maintain [0,1] range
        if self.norm == 'inf':
            delta = torch.clamp(perturbed_image - orig_image, min=-1 * self.epsilon, max=self.epsilon)
        elif self.norm == 'two':
            delta = perturbed_image - orig_image
            delta_norms = torch.norm(delta.view(self.batch_size, -1), p=2, dim=1)
            factor = self.epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)
        perturbed_image = torch.clamp(orig_image + delta, 0, 1)
        # Return the perturbed image
        return perturbed_image

    def fit(self, val_loader):
        """
        Fit the model using the given loaders for the given number
        of epochs.

        Parameters
        ----------
        train_loader :
        val_loader :
        """
        # attributes
        self.train_losses_ = torch.zeros(self.epochs)
        self.val_losses_ = torch.zeros(self.epochs)
        # ---- train process ----
        for epoch in trange(1, self.epochs + 1, desc='Traning Model on {} epochs'.format(self.epochs)):
            # train
            get_score = True
            if not self.mode == 'adv_attack' and get_score:
                # validate
                self.logger.info("start test...")
                self._evaluate(val_loader, epoch, self.model)

            if self.mode == 'adv_attack':
                self.adv_attack(val_loader, epoch)

            if get_score:
                score = self.metrics.get_results()
                if self.logger != None:
                    string = "epoch: " + str(epoch) + "   "
                    for item in score:
                        string += item + ": {}    ".format(score[item])
                    self.logger.info(string)
                if self.mode == 'adv_attack' or self.mode == 'test' or self.mode == 'trans_test':
                    break
                if score["Mean IoU"] > self.mIoU:
                    self.mIoU = score["Mean IoU"]
                    self.save_ckpt(epoch)



    @torch.inference_mode()
    def trans_test(self, data_loader, model=None, model_name=None):
        self.metrics.reset()
        model.cuda()
        model.eval()
        pred_path = os.path.join(self.results_path, '{}'.format(model_name))


        with tqdm(data_loader, unit=" validating-batch", colour="green") as evaluation:
            for i, (images, labels, name) in enumerate(evaluation):
                name = name[0]
                evaluation.set_description(f"Validation")
                images = self.attack_imgs[i].to(self.device)
                labels =  labels.to(self.device)

                preds = model(images)

                self.metrics.update(labels.detach().cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy())

                lp = torch.argmax(preds, dim=1)[0]

                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)

                if self.data_name == 'pascal_voc':
                    Image.fromarray(data_loader.dataset.decode_target(lp.cpu().numpy())).save(f"{pred_path}/{name}")
                elif self.data_name == 'citys':
                    city = name.split('_')[0]
                    if not os.path.exists(os.path.join(pred_path, city)):
                        os.makedirs(os.path.join(pred_path, city))
                    pred = data_loader.dataset.decode_target(lp).astype(np.uint8)
                    Image.fromarray(pred).save(f"{pred_path}/{city}/{name}.png")

        score = self.metrics.get_results()
        if self.logger != None:
            string = f"\n{model_name}\n"
            for item in score:
                string += item + ": {}    ".format(score[item])
            self.logger.info(string)



    @torch.inference_mode()
    def _evaluate(self, data_loader, epoch=None, model=None):
        self.metrics.reset()
        model.cuda()
        model.eval()
        pred_path = os.path.join(self.results_path, '{}'.format(self.target_model))

        with tqdm(data_loader, unit=" validating-batch", colour="green") as evaluation:
            for i, (images, labels, name) in enumerate(evaluation):
                name = name[0]
                evaluation.set_description(f"Validation")
                images, labels = images.to(self.device), labels.to(self.device)
                if len(self.attack_imgs) != 0:
                    images = self.attack_imgs[i]

                preds = model(images)

                self.metrics.update(labels.detach().cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy())

                lp = torch.argmax(preds, dim=1)[0]

                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)

                if self.data_name == 'pascal_voc':
                    Image.fromarray(data_loader.dataset.decode_target(lp.cpu().numpy())).save(f"{pred_path}/{name}")
                elif self.data_name == 'citys':
                    city = name.split('_')[0]
                    if not os.path.exists(os.path.join(pred_path, city)):
                        os.makedirs(os.path.join(pred_path, city))
                    pred = data_loader.dataset.decode_target(lp).astype(np.uint8)
                    Image.fromarray(pred).save(f"{pred_path}/{city}/{name}.png")

        score = self.metrics.get_results()
        if self.logger != None:
            string = f"{self.target_model}\n"
            for item in score:
                string += item + ": {}    ".format(score[item])
            self.logger.info(string)


    @torch.enable_grad()
    def adv_attack(self, data_loader, epoch):
        self.model.eval()
        self.metrics.reset()
        if self.targeted:
            self.actual_metrics.reset()
            self.initial_metrics.reset()
        losses = torch.zeros(len(data_loader))
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        eps = [0.03 / i for i in std]
        clamp_min = [-m / s for m, s in zip(mean, std)]
        clamp_max = [(1 - m) / s for m, s in zip(mean, std)]

        with tqdm(data_loader, unit=" validating-batch", colour="green") as evaluation:
            for i, (images, labels, name) in enumerate(evaluation):
                name = name[0]
                evaluation.set_description(f"Validation")
                images, labels = images.to(self.device), labels.to(self.device)
                orig_labels = labels.clone()
                if self.targeted:
                    labels = torch.ones_like(labels)
                orig_image = images.clone()

                def get_source_layer(model):
                    layers = []
                    layer_text = self.source_layer.split('_')

                    model_backbone = model.pretrained

                    if len(layer_text) == 1:  # only layer
                        layers.append(getattr(model_backbone, layer_text[0]))
                    elif len(layer_text) == 2:
                        layers.append(getattr(model_backbone, layer_text[0])[int(layer_text[1])])
                    elif len(layer_text) == 3:
                        layers.append(getattr(getattr(model_backbone, layer_text[0])[int(layer_text[1])], layer_text[2]))

                    return layers

                def get_mid_output(m, i, o):
                    global mid_output
                    mid_output = o

                if self.source_layer is not None:
                    feature_layer = get_source_layer(self.model)[0]
                    h = feature_layer.register_forward_hook(get_mid_output)  # layer select

                # with torch.no_grad():
                orig_preds = self.model(images)
                if 'fspgd' in self.attack:
                    mid_original = torch.zeros(mid_output.size()).cuda()
                    mid_original.copy_(mid_output)
                    mid_original = torch.nn.functional.normalize(mid_original, dim=1)  # original feature map normalize

                if 'fspgd' in self.attack:
                    if self.norm == 'inf':
                        images = functions.init_linf(
                            images,
                            epsilon=eps,
                            clamp_min=clamp_min,
                            clamp_max=clamp_max
                        )
                    elif self.norm == 'two':
                        images = functions.init_l2(
                            images,
                            epsilon=eps,
                            clamp_min=clamp_min,
                            clamp_max=clamp_max
                        )

                # images.retain_grad()
                images.requires_grad = True
                preds = self.model(images)
                if 'fspgd' in self.attack:
                    mid_adv = torch.zeros(mid_output.size()).cuda()
                    mid_adv.copy_(mid_output)
                    mid_adv = torch.nn.functional.normalize(mid_adv, dim=1)  # adversarial attack feature map

                if "CrossEntropyLoss" in str(type(self.criterion)):
                    loss = self.criterion(preds.float(), labels.long())
                else:
                    loss = self.criterion(preds.float(), labels.float())
                # -----------------------------------------------------
                for t in range(self.iterations):
                    if self.attack == 'fspgd':
                        loss = functions.fspgd(
                            mid_original=mid_original,
                            mid_adv=mid_adv,
                            cosine=self.cosine,
                            iteration=t,
                            iterations=self.iterations,
                        )
                        loss.backward(retain_graph=True)


                    if self.norm == 'inf':
                        images = functions.step_inf(
                            perturbed_image=images,
                            epsilon=eps,
                            data_grad=images.grad,
                            orig_image=orig_image,
                            alpha=self.alpha,
                            targeted=self.targeted,
                            clamp_min=clamp_min,
                            clamp_max=clamp_max,
                            grad_scale=None
                        )
                    elif self.norm == 'two':
                        images = functions.step_l2(
                            perturbed_image=images,
                            epsilon=eps,
                            data_grad=images.grad,
                            orig_image=orig_image,
                            alpha=self.alpha,
                            targeted=self.targeted,
                            clamp_min=clamp_min,
                            clamp_max=clamp_max,
                            grad_scale=None
                        )

                    images.requires_grad = True
                    preds = self.model(images)
                    if 'fspgd' in self.attack:
                        mid_adv = torch.zeros(mid_output.size()).cuda()
                        mid_adv.copy_(mid_output)
                        mid_adv = torch.nn.functional.normalize(mid_adv, dim=1)  # adversarial attack feature map

                    if "CrossEntropyLoss" in str(type(self.criterion)):
                        loss = self.criterion(preds.float(), labels.long())
                    else:
                        loss = self.criterion(preds.float(), labels.float())

                if 'fspgd' in self.attack:
                    h.remove()
                # ----------------------------------------------------
                self.attack_imgs.append(images)
                adv_img = images[0].detach().cpu().numpy()
                for c, (mean_c, std_c) in enumerate(zip(mean, std)):
                    adv_img[c, :, :] *= std_c
                    adv_img[c, :, :] += mean_c
                adv_img = adv_img * 255
                adv_img = adv_img.transpose(1, 2, 0)
                adv_img = adv_img.astype(np.uint8)
                if not os.path.exists(os.path.join(self.results_path, 'example')):
                    os.makedirs(os.path.join(self.results_path, 'example'))
                Image.fromarray(adv_img).save(os.path.join(self.results_path, 'example/%s' % name))
                if self.data_name == 'pascal_voc':
                    Image.fromarray(adv_img).save(os.path.join(self.results_path, 'example/%s' % name))
                elif self.data_name == 'citys':
                    city = name.split('_')[0]
                    city_exp = os.path.join(self.results_path, 'example', city)
                    if not os.path.exists(city_exp):
                        os.makedirs(city_exp)
                    Image.fromarray(adv_img).save(os.path.join(city_exp, '%s' % name))

                self.metrics.update(labels.detach().cpu().numpy(), preds.detach().max(dim=1)[1].cpu().numpy())

                if self.targeted:
                    self.actual_metrics.update(orig_labels.detach().cpu().numpy(),
                                               preds.detach().max(dim=1)[1].cpu().numpy())
                    self.initial_metrics.update(orig_preds.detach().max(dim=1)[1].cpu().numpy(),
                                                preds.detach().max(dim=1)[1].cpu().numpy())
                self.val_losses_[epoch - 1] = loss.item()
                evaluation.set_postfix(loss=loss.item())
                losses[i] = loss.item()

    def _get_device(self, _device):
        if _device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {device}"
            warnings.warn(msg)
            return device
        return _device

