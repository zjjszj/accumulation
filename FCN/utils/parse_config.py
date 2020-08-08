import os


def parse_model_cfg(cfg, model_cfg):
    """
    Args:
        cfg: defaults cfg.
        model_cfg: model cfg.
    """
    if not model_cfg.endswith('cfg'):
        model_cfg=model_cfg+'.cfg'
    if not os.path.exists(model_cfg) and os.path.exists('FCN/configs' + os.sep + model_cfg):  # add cfg/ prefix if omitted
        path = 'FCN/configs' + os.sep + model_cfg

    with open(model_cfg, 'r') as f:
        lines=f.read().split('\n')  # ['aaaa', 'bbb']

    lines=[x.lstrip().rstrip() for x in lines]  # get rid of fringe space
    mdefs=[]
    for line in lines:
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            mdefs.append({})
            mdefs[-1]['type']=line[1:-1]
            if line[1:-1]=='convolutional':
                mdefs[-1]['batch_normalize']=0     # initialize
        else:
            key, value=line.split('=')
            key=key.rstrip()
            # recovery to value oril type.
                # could return array
            value=value.split(',')
            if len(value)==1:
                if value[0].isnumeric():  # return integer
                    mdefs[-1][key]=int(value[0])
                elif value[0].find('.')!=-1:  # return float
                    mdefs[-1][key] = float(value[0])
                else:
                    value=value[0].strip()
                    mdefs[-1][key]=value  # return string
            else:
                if value[0].isnumeric():  # return integer
                    mdefs[-1][key]=[int(x) for x in value]
                elif value[0].find('.')!=-1:  # return float
                    mdefs[-1][key] = [float(x) for x in value]
                else:
                    mdefs[-1][key]=value  # return string

    # Check all fields are supported
    supported = set(cfg.SOLVER.SUPPORT_KEY)     # set
    used_key=set()
    for item in mdefs[1:]:
        for k in item:
            used_key.add(k)
    assert supported == used_key, "supported key and used_key are different. the not repeat ele: %s"%set.symmetric_difference(supported, used_key)
    return mdefs


if __name__ == '__main__':
    from FCN.config import cfg
    mdefs=parse_model_cfg(cfg, '../configs/vgg16-fcn32s.cfg')
    print(mdefs)







