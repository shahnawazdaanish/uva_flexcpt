import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from abc import ABC, abstractmethod
from src.soed.dynamic_gp import DynamicGP

class AnimatedPlot(ABC):    
    def __init__(self):
            self.fig = None
            self.ani = None

    @abstractmethod
    def update(self, frame):
        pass
    
    @abstractmethod
    def animate(self, num_frames):
        pass



class PlotlyAnimatedPlot(AnimatedPlot):
    def __init__(self):

        self.fig = None
        pass
    
    def update(self, frame):
        pass
    
    def animate(self, num_frames):
        pass


class MultiOutputSliceAnimator:
    def __init__(self, agent, optimal_path, **kwargs):
        self.agent = agent
        self.path = optimal_path
        self.kwargs = kwargs

    def animate(self, **kwargs):
        for i in range(self.agent.m):
            print(f"\n Animating output {i}")
            anim = SliceMeanVarianceAnimatedPlot(
                self.agent,
                self.path,
                output_idx=i,
                **self.kwargs
            )
            anim.animate(**kwargs)


class SliceMeanVarianceAnimatedPlot(AnimatedPlot):
    def __init__(
        self,
        agent,
        optimal_path,
        output_idx=0,
        dim_x=0,
        dim_y=1,
        fixed_at="current",
        res=40,
    ):
        super().__init__()

        self.agent = agent
        self.path = optimal_path
        self.output_idx = output_idx
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.res = res
        self.num_frames = optimal_path.shape[0]
        self.paused = False

        self.bounds = agent.bounds
        self.model = agent.models[output_idx]
        self.noise = agent.likelihoods[output_idx].noise.item()

        self.fixed = self.get_fixed_point(agent, fixed_at)
        self.ls_eff = self.effective_lengthscales(agent, mode="min")

        self._prepare_grid()
        self._prepare_gp()
        self._setup_figure()
        self._setup_sliders()

    def effective_lengthscales(self, agent, mode="min"):
        lss = torch.stack([
            m.covar_module.base_kernel.lengthscale.squeeze().detach()
            for m in agent.models
        ])
        if mode == "min":
            return torch.min(lss, dim=0).values
        elif mode == "mean":
            return lss.mean(dim=0)
        else:
            raise ValueError("mode must be 'min' or 'mean'")
        

    def predictive_variance_after_k_steps(self, k):
        X_aug = torch.cat([self.agent.X, self.path[:k+1]])
        Y_aug = torch.cat([
            self.agent.Y[:, self.output_idx: self.output_idx+1],
            torch.zeros((k+1, 1), dtype=torch.float64)
        ])

        with torch.no_grad():
            temp_model = DynamicGP(
                X_aug, Y_aug,
                self.agent.likelihoods[self.output_idx],
                self.agent.bounds
            )
            
            post = temp_model.posterior(self.grid)
            return post.variance.view(self.res, self.res).cpu().numpy()
        
    
    def inverse_transform_mean(self, model, mean_tensor):
        ot = model.outcome_transform
        if ot is None:
            return mean_tensor
        mean_phys, _ = ot.untransform(mean_tensor)
        return mean_phys



    # --------------------------------------------------
    def get_fixed_point(self, agent, mode="current"):
        if mode == "current":
            return agent.X[-1].clone()
        elif mode == "mean":
            return agent.X.mean(dim=0)
        else:
            raise ValueError("fixed_at must be 'current' or 'mean'")

    # --------------------------------------------------
    def _prepare_grid(self):
        self.x = torch.linspace(self.bounds[0, self.dim_x],
                                self.bounds[1, self.dim_x], self.res)
        self.y = torch.linspace(self.bounds[0, self.dim_y],
                                self.bounds[1, self.dim_y], self.res)

        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing="xy")

        grid = self.fixed.repeat(self.res * self.res, 1)
        grid[:, self.dim_x] = self.X.flatten()
        grid[:, self.dim_y] = self.Y.flatten()
        self.grid = grid

    # --------------------------------------------------
    def _prepare_gp(self):
        with torch.no_grad():
            post = self.model.posterior(self.grid)

            mean_phys = self.inverse_transform_mean(self.model, post.mean)
            self.Z_mean = mean_phys.view(self.res, self.res).cpu().numpy()

            # self.Z_mean = post.mean.view(self.res, self.res).cpu().numpy()
            self.Z_var = post.variance.view(self.res, self.res).cpu().numpy()

    # --------------------------------------------------
    def _setup_figure(self):
        self.fig = plt.figure(figsize=(14, 6))

        self.ax_mean = self.fig.add_subplot(121, projection="3d")
        self.ax_var = self.fig.add_subplot(122, projection="3d")

        X_obs = self.agent.X
        Y_obs = self.agent.Y[:, self.output_idx]

        xs = X_obs[:, self.dim_x].cpu().numpy()
        ys = X_obs[:, self.dim_y].cpu().numpy()

        # inverse‑transform observations if needed
        ys_val = self.inverse_transform_mean(self.model, Y_obs.unsqueeze(-1)).cpu().numpy()

        self.ax_mean.scatter(
            xs, ys, ys_val,
            c="black", s=20, alpha=0.8, label="Observed data"
        )

        self.ax_mean.set_title("Predictive Mean")
        self.ax_var.set_title("Predictive Variance")

        self.mean_surf = self.ax_mean.plot_surface(
            self.X.numpy(), self.Y.numpy(), self.Z_mean,
            cmap="viridis", alpha=0.7
        )

        self.var_surf = self.ax_var.plot_surface(
            self.X.numpy(), self.Y.numpy(), self.Z_var,
            cmap="plasma", alpha=0.8
        )

        self.ax_mean.legend()

        # Lengthscale-aware aspect ratio
        for ax in [self.ax_mean, self.ax_var]:
            ax.set_box_aspect([
                1.0 / self.ls_eff[self.dim_x].item(),
                1.0 / self.ls_eff[self.dim_y].item(),
                1.0
            ])
            ax.set_xlabel(f"Input {self.dim_x}")
            ax.set_ylabel(f"Input {self.dim_y}")
            ax.set_zlabel("Value")

        # Path lines
        self.path_mean, = self.ax_mean.plot([], [], [], "o-", color="orange")
        self.path_var, = self.ax_var.plot([], [], [], "o-", color="orange")

    # --------------------------------------------------
    def _setup_sliders(self):
        ax_dx = plt.axes([0.25, 0.02, 0.5, 0.02])
        ax_dy = plt.axes([0.25, 0.05, 0.5, 0.02])

        self.slider_dx = Slider(ax_dx, "dim_x", 0, self.agent.d - 1,
                                valinit=self.dim_x, valstep=1)
        self.slider_dy = Slider(ax_dy, "dim_y", 0, self.agent.d - 1,
                                valinit=self.dim_y, valstep=1)

        self.slider_dx.on_changed(self._on_slider_change)
        self.slider_dy.on_changed(self._on_slider_change)

    # --------------------------------------------------
    def _on_slider_change(self, _):
        self.paused = True

        self.dim_x = int(self.slider_dx.val)
        self.dim_y = int(self.slider_dy.val)

        self._prepare_grid()
        self._prepare_gp()

        # Remove old surfaces safely
        self.mean_surf.remove()
        self.var_surf.remove()

        self.mean_surf = self.ax_mean.plot_surface(
            self.X.numpy(), self.Y.numpy(), self.Z_mean,
            cmap="viridis", alpha=0.7
        )

        self.var_surf = self.ax_var.plot_surface(
            self.X.numpy(), self.Y.numpy(), self.Z_var,
            cmap="plasma", alpha=0.8
        )

        self.fig.canvas.draw_idle()
        self.paused = False

    # --------------------------------------------------
    def update(self, frame):
        if self.paused:
            return self.path_mean, self.path_var

        p = self.path[:frame + 1]

        xs = p[:, self.dim_x].cpu().numpy()
        ys = p[:, self.dim_y].cpu().numpy()

        
        self.Z_var = self.predictive_variance_after_k_steps(frame)
        zs = np.full_like(xs, self.Z_var.max())

        # zs = np.full_like(xs, self.Z_var.max() + self.noise)

        self.path_mean.set_data(xs, ys)
        self.path_mean.set_3d_properties(zs)

        self.path_var.set_data(xs, ys)
        self.path_var.set_3d_properties(zs)

        return self.path_mean, self.path_var

    # --------------------------------------------------
    def animate(self, num_frames=None, save_path=None, fps=2):
        if num_frames is None:
            num_frames = self.num_frames

        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=num_frames,
            interval=1000 // fps,
            blit=False
        )

        if save_path:
            if save_path.endswith(".gif"):
                self.ani.save(save_path, writer="pillow", fps=fps)
            elif save_path.endswith(".mp4"):
                self.ani.save(save_path, writer="ffmpeg", fps=fps)

        plt.show(block=True)




class OptimizationAnimation(AnimatedPlot):
    def __init__(self, model, bounds, input_features, output_feature):
        self.model = model
        self.bounds = bounds
        self.input_features = input_features
        self.output_feature = output_feature
        self.fig, self.ax = plt.subplots()
    
    def update(self, frame):
        # This method will be called for each frame of the animation
        # You can use the model to predict the output for the current input and update the plot accordingly
        pass
    
    def animate(self, num_frames):
        ani = animation.FuncAnimation(self.fig, self.update, frames=num_frames, repeat=False)
        plt.show()