import torch


class EngineeringConstraints:
    """Explicit geometry-based engineering constraint layer for the planner.

    This version separates the constraint logic from the GP/agent logic so it can be
    reused, tested, and replaced without editing the original planner.
    """

    def __init__(
        self,
        feature_names,
        mass1_name="Mass1",
        mass2_name="Mass2",
        boost_name="Boost pressure",
        ivo_name="IVO",
        ivc_name="IVC",
        evo_name="EVO",
        evc_name="EVC",
        load_limit=30.0,
        boost_slope=0.0922,
        boost_intercept=0.8378,
        boost_band=0.5,
        min_load=3.0,
        ambient_pressure=1.0,
        tc_boost_limit=3.8,
        enable_br_limit=True,
        enable_vva_limit=True,
    ):
        self.feature_names = list(feature_names)
        self.idxs = {
            name: self.feature_names.index(feat)
            for name, feat in zip(
                ["m1", "m2", "bst", "ivo", "ivc", "evo", "evc"],
                [mass1_name, mass2_name, boost_name, ivo_name, ivc_name, evo_name, evc_name],
            )
        }
        self.load_limit = float(load_limit)
        self.boost_slope = float(boost_slope)
        self.boost_intercept = float(boost_intercept)
        self.boost_band = float(boost_band)
        self.min_load = float(min_load)
        self.ambient_pressure = float(ambient_pressure)
        self.tc_boost_limit = float(tc_boost_limit)
        self.enable_br_limit = bool(enable_br_limit)
        self.enable_vva_limit = bool(enable_vva_limit)

    def _boost_band(self, mass_sum):
        return self.boost_slope * mass_sum + self.boost_intercept

    def _vva_bounds(self, bin_idx, dev):
        limits = [
            (self.idxs["ivo"], torch.tensor([350.0, 330.0, 345.0], device=dev), torch.tensor([435.0, 390.0, 365.0], device=dev)),
            (self.idxs["ivc"], torch.tensor([500.0, 500.0, 495.0], device=dev), torch.tensor([540.0, 570.0, 535.0], device=dev)),
            (self.idxs["evo"], torch.tensor([128.0, 128.0, 128.0], device=dev), torch.tensor([218.0, 218.0, 218.0], device=dev)),
            (self.idxs["evc"], torch.tensor([270.0, 330.0, 345.0], device=dev), torch.tensor([350.0, 370.0, 355.0], device=dev)),
        ]
        return [(idx, low[bin_idx], high[bin_idx]) for idx, low, high in limits]

    def _to_original_values(self, x_scaled, scaler_x=None):
        if scaler_x is None:
            return x_scaled

        if hasattr(scaler_x, 'scaler'):
            scaler_obj = scaler_x.scaler
        else:
            scaler_obj = scaler_x

        if not hasattr(scaler_obj, 'mean_') or not hasattr(scaler_obj, 'scale_'):
            return x_scaled

        mean = torch.as_tensor(scaler_obj.mean_, dtype=x_scaled.dtype, device=x_scaled.device)
        scale = torch.as_tensor(scaler_obj.scale_, dtype=x_scaled.dtype, device=x_scaled.device)
        return x_scaled * scale + mean

    def feasible_mask(self, x):
        """Return a boolean mask for the feasible polygon region."""
        m1 = x[..., self.idxs["m1"]]
        m2 = x[..., self.idxs["m2"]]
        bst = x[..., self.idxs["bst"]]
        mass_sum = m1 + m2
        lower_boost = self._boost_band(mass_sum) - self.boost_band
        upper_boost = self._boost_band(mass_sum) + self.boost_band

        ok = (
            (m1 < self.load_limit)
            & (mass_sum < self.load_limit)
            & (mass_sum >= self.min_load)
            & (bst <= self.tc_boost_limit)
            & (bst >= self.ambient_pressure)
            & (bst >= lower_boost)
            & (bst <= upper_boost)
        )

        b = torch.clamp((m1 // 10).long(), 0, 2)
        dev = m1.device

        if self.enable_br_limit:
            br_low = torch.tensor([0.5, 0.9, 0.0], device=dev)[b]
            br_high = torch.tensor([3.5, 3.0, 1.5], device=dev)[b]
            ok &= (m2 > br_low) & (m2 < br_high)

        if self.enable_vva_limit:
            for idx, low, high in self._vva_bounds(b, dev):
                ok &= (x[..., idx] >= low) & (x[..., idx] <= high)

        return ok.all(dim=-1)

    def feasible_mask_scaled(self, x_scaled, scaler_x=None):
        x_orig = self._to_original_values(x_scaled, scaler_x)
        return self.feasible_mask(x_orig)

    def constraint_penalty(self, x):
        """Soft penalty used for optimization fallback values."""
        m1 = x[..., self.idxs["m1"]]
        m2 = x[..., self.idxs["m2"]]
        bst = x[..., self.idxs["bst"]]
        mass_sum = m1 + m2
        corridor_center = self._boost_band(mass_sum)
        lower_boost = corridor_center - self.boost_band
        upper_boost = corridor_center + self.boost_band

        pen = (
            torch.relu(m1 - self.load_limit).pow(2)
            + torch.relu(mass_sum - self.load_limit).pow(2)
            + torch.relu(self.min_load - mass_sum).pow(2)
            + torch.relu(bst - self.tc_boost_limit).pow(2)
            + torch.relu(self.ambient_pressure - bst).pow(2)
            + torch.relu(lower_boost - bst).pow(2)
            + torch.relu(bst - upper_boost).pow(2)
        )

        boost_margin = 0.15 * self.boost_band
        load_margin = max(0.25, 0.05 * self.load_limit)
        pen += torch.relu(boost_margin - (bst - lower_boost)).pow(2)
        pen += torch.relu(boost_margin - (upper_boost - bst)).pow(2)
        pen += torch.relu(self.min_load + load_margin - mass_sum).pow(2)
        pen += torch.relu(m1 - (self.load_limit - load_margin)).pow(2)
        pen += torch.relu(mass_sum - (self.load_limit - load_margin)).pow(2)

        b = torch.clamp((m1 // 10).long(), 0, 2)
        dev = m1.device
        if self.enable_br_limit:
            br_low = torch.tensor([0.5, 0.9, 0.0], device=dev)[b]
            br_high = torch.tensor([3.5, 3.0, 1.5], device=dev)[b]
            pen += torch.relu(br_low - m2).pow(2) + torch.relu(m2 - br_high).pow(2)

        if self.enable_vva_limit:
            for idx, low, high in self._vva_bounds(b, dev):
                pen += torch.relu(low - x[..., idx]).pow(2) + torch.relu(x[..., idx] - high).pow(2)

        return pen.sum(dim=-1)

    def constraint_penalty_scaled(self, x_scaled, scaler_x=None):
        x_orig = self._to_original_values(x_scaled, scaler_x)
        return self.constraint_penalty(x_orig)

    def interior_margin_penalty(self, x):
        """Preference for staying in the interior of the feasible region."""
        m1 = x[..., self.idxs["m1"]]
        m2 = x[..., self.idxs["m2"]]
        bst = x[..., self.idxs["bst"]]
        mass_sum = m1 + m2
        corridor_center = self._boost_band(mass_sum)
        lower_boost = corridor_center - self.boost_band
        upper_boost = corridor_center + self.boost_band

        boost_margin = 0.15 * self.boost_band
        load_margin = max(0.25, 0.05 * self.load_limit)
        return (
            torch.relu(boost_margin - (bst - lower_boost)).pow(2)
            + torch.relu(boost_margin - (upper_boost - bst)).pow(2)
            + torch.relu(self.min_load + load_margin - mass_sum).pow(2)
            + torch.relu(m1 - (self.load_limit - load_margin)).pow(2)
            + torch.relu(mass_sum - (self.load_limit - load_margin)).pow(2)
        ).sum(dim=-1)

    def interior_margin_penalty_scaled(self, x_scaled, scaler_x=None):
        x_orig = self._to_original_values(x_scaled, scaler_x)
        return self.interior_margin_penalty(x_orig)
