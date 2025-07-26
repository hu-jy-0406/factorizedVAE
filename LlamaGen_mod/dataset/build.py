from LlamaGen_mod.dataset.imagenet import build_imagenet, build_imagenet_code
from LlamaGen_mod.dataset.cifar10 import build_cifar10, build_cifar10_code
from LlamaGen_mod.dataset.coco import build_coco
from LlamaGen_mod.dataset.openimage import build_openimage
from LlamaGen_mod.dataset.pexels import build_pexels
from LlamaGen_mod.dataset.t2i import build_t2i, build_t2i_code, build_t2i_image


def build_dataset(type, args, **kwargs):
    # images
    print(f'Building dataset {args.dataset} ...')
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'cifar10':
        return build_cifar10(args, **kwargs)
    if args.dataset == 'cifar10_code':
        return build_cifar10_code(type, args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        return build_t2i_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')