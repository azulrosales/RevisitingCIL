def get_model(model_name, args):
    name = model_name.lower()
    if name=="aper_bn":
        from models.aper_bn import Learner
        return Learner(args)
    else:
        assert 0
