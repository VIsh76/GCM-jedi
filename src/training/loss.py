import torch

def Loss(x, y):
    return torch.nn.MSELoss(x, y)

class WeightMSE(torch.nn.MSELoss):
    def __init__(self, 
                 surface_weigth:torch.tensor, 
                 column_weights:torch.tensor, 
                 level_weights:torch.tensor, 
                 lat_weights:torch.tensor, 
                 regional_mask:torch.tensor=None,
                 size_average=None, 
                 reduce=None, 
                 reduction: str = 'mean') -> None:
        super(WeightMSE, self).__init__(size_average, reduce, reduction)
        self.surface_weigths = surface_weigth  # Must have size : (1, 1, 1, n_sur_vars)
        self.column_weights = column_weights  # Must have size : (1, 1, 1, n_col_vars)
        self.level_weights = level_weights    # Must have size : (1, 1, 1, lev, 1)
        self.lat_weights_sur = lat_weights    # Must have size : (1, lat, 1, 1)
        self.lat_weights_col = torch.unsqueeze(lat_weights, axis=-1) # Will be size (1, lat, 1, 1, 1)
        if regional_mask is None:
            self.regional_mask_col = torch.ones_like(self.column_weights)
            self.regional_mask_sur = torch.ones_like(self.surface_weigths)
        else:
            self.regional_mask_sur = torch.clone(regional_mask)
            self.regional_mask_col = torch.clone(torch.unsqueeze(regional_mask, axis=-1))
                

    def __call__(self, pred_col, target_col, pred_sur, target_sur):        
        # Column pred
        delta_col = self.regional_mask_col * self.column_weights * self.lat_weights_col * self.level_weights *(target_col - pred_col)
        L_col =  super(WeightMSE, self).__call__(torch.zeros_like(target_col), delta_col)
        # Surface pred
        delta_sur = self.regional_mask_sur * self.surface_weigths * self.lat_weights_sur * (target_sur - pred_sur)
        L_sur = super(WeightMSE, self).__call__(torch.zeros_like(target_sur), delta_sur)
#        L_col =  super(WeightMSE, self).__call__(pred_col, target_col)
#        # Surface pred
#        L_sur = super(WeightMSE, self).__call__(pred_sur, target_sur)
        return L_col, L_sur
