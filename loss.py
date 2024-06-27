import torch 


def wasserstein_loss(real_output, fake_output):
    return -torch.mean(real_output) + torch.mean(fake_output)


def GradientPenalty(discriminator, real_samples, fake_samples, text_embeddings):
    alpha = torch.randn(real_samples.size(0),1,1,1,device=real_samples.device)
    interpolates  = (alpha*real_samples+(1-alpha)*fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates, text_embeddings)
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs = interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean()
    return gradient_penalty


