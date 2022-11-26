import torch


class LagrangianOptimization:

    min_alpha = None
    max_alpha = None
    device = None
    original_optimizer = None
    gradient_accumulation_steps = None
    update_counter = 0

    def __init__(self, original_optimizer, device, init_alpha=5, min_alpha=-2, max_alpha=30, alpha_optimizer_lr=1e-2, gradient_accumulation_steps=None, max_grad_norm=None):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.update_counter = 0

        self.alpha = torch.tensor(init_alpha, dtype=torch.float, device=device, requires_grad=True)
        self.optimizer_alpha = torch.optim.RMSprop([self.alpha], lr=alpha_optimizer_lr, centered=True)
        self.original_optimizer = original_optimizer
        self.max_grad_norm = max_grad_norm

    def update(self, f, g, model):
        """
        L(x, lambda) = f(x) + lambda g(x)

        :param f_function:
        :param g_function:
        :return:
        """

        loss = f + torch.nn.functional.softplus(self.alpha) * g
        if isinstance(model, torch.nn.DataParallel):   
            loss = loss.mean()
        try:
            #with torch.autograd.detect_anomaly():
            loss.backward()
        except RuntimeError:
            import pdb; pdb.set_trace()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        
        if self.gradient_accumulation_steps is not None and self.gradient_accumulation_steps > 1:
            if self.update_counter % self.gradient_accumulation_steps == 0:
                self.original_optimizer.step()
                self.alpha.grad *= -1
                #print(self.alpha, self.alpha.grad)
                self.optimizer_alpha.step()
                self.original_optimizer.zero_grad()
                self.optimizer_alpha.zero_grad()
            
            self.update_counter += 1                
        else: 
            self.original_optimizer.step()
            self.alpha.grad *= -1
            self.optimizer_alpha.step()
            self.original_optimizer.zero_grad()
            self.optimizer_alpha.zero_grad()

        if self.alpha.item() < self.min_alpha:
            self.alpha.data = torch.full_like(self.alpha.data, self.min_alpha)
        elif self.alpha.item() > self.max_alpha:
            self.alpha.data = torch.full_like(self.alpha.data, self.max_alpha)
        
        return self.alpha.item()
