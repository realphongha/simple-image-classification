import torch
import collections


def load_checkpoint(model, checkpoint, strict=False):
    checkpoint = torch.load(checkpoint,
                            map_location=lambda storage, loc: storage)
    checkpoint = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    source_state_ = checkpoint
    source_state = {}

    target_state = model.state_dict()
    new_target_state = collections.OrderedDict()

    for k in source_state_:
        if k.startswith('module') and not k.startswith('module_list'):
            new_k = k[7:]
            if new_k.startswith('backbone'):
                new_k = new_k[9:]
            source_state[new_k] = source_state_[k]
        else:
            new_k = k
            if new_k.startswith('backbone'):
                new_k = new_k[9:]
            source_state[new_k] = source_state_[k]

    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    model.load_state_dict(new_target_state, strict=strict)

    return model
    