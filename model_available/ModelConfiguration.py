class MMSegmentationConfig:
    """
    Create class with following fixed function names and the number of arguents to train your model from external source
    """

    try:
        import torch
    except:
        pass

    def get_model(self, data, backbone=None, **kwargs):
        from arcgis.learn.models._mmlab_utils import mmlab_models, prepare_mmbatch

        model_name = kwargs.get("model")
        if model_name.startswith("prithvi100m"):
            # register custom prithvi head
            from arcgis.learn.models._prithvi_archs import TemporalViTEncoder

        kwargs["model_type"] = "Segmentation"
        model, cfg = mmlab_models(data, **kwargs)
        model._is_transformer = kwargs.get("is_transformer", False)
        self.model = model
        self.cfg = cfg
        self.prepare_mmbatch = prepare_mmbatch
        return model

    def on_batch_begin(self, learn, model_input_batch, model_target_batch, **kwargs):
        if kwargs.get("train"):
            self.model.train_val = False
        else:
            self.model.train_val = True
        learn.model.train()
        batch_shape = model_input_batch.permute(0, 2, 3, 1).shape
        gt_batch = []
        for gt_sem_seg in model_target_batch:
            data_sample = self.prepare_mmbatch(
                batch_shape, gt_sem_seg=gt_sem_seg, model_type="Segmentation"
            )
            gt_batch.append(data_sample)

        # handle batch size one in training
        if model_input_batch.shape[0] < 2:
            model_input_batch = self.torch.cat((model_input_batch, model_input_batch))
            gt_batch.append(gt_batch[0])

        model_input = [model_input_batch, gt_batch]
        return model_input, model_target_batch

    def transform_input(self, xb):
        batch_shape = xb.permute(0, 2, 3, 1).shape
        img_metas = []
        for _ in range(xb.shape[0]):
            data_sample = self.prepare_mmbatch(batch_shape, model_type="Segmentation")
            img_metas.append(data_sample)

        model_input = [xb, img_metas]
        return model_input

    def transform_input_multispectral(self, xb):
        return self.transform_input(xb)

    def loss(self, model_output, *model_target):
        return model_output[1]

    def post_process(self, pred, thres=0.5, thinning=True, prob_raster=False):
        """
        In this function you have to return list with appended output for each image in the batch with shape [C=1,H,W]!
        """
        if prob_raster:
            return pred
        else:
            pred = self.torch.unsqueeze(pred.argmax(dim=1), dim=1)
        return pred
