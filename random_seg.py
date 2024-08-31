import numpy as np
import cv2
from tqdm import tqdm

import types
from lime.utils.generic_utils import has_arg
from skimage.segmentation import felzenszwalb, slic, quickshift


class BaseWrapper(object):
    """Base class for LIME Scikit-Image wrapper


    Args:
        target_fn: callable function or class instance
        target_params: dict, parameters to pass to the target_fn


    'target_params' takes parameters required to instanciate the
        desired Scikit-Image class/model
    """

    def __init__(self, target_fn=None, **target_params):
        self.target_fn = target_fn
        self.target_params = target_params

    def _check_params(self, parameters):
        """Checks for mistakes in 'parameters'

        Args :
            parameters: dict, parameters to be checked

        Raises :
            ValueError: if any parameter is not a valid argument for the target function
                or the target function is not defined
            TypeError: if argument parameters is not iterable
         """
        a_valid_fn = []
        if self.target_fn is None:
            if callable(self):
                a_valid_fn.append(self.__call__)
            else:
                raise TypeError('invalid argument: tested object is not callable,\
                 please provide a valid target_fn')
        elif isinstance(self.target_fn, types.FunctionType) \
                or isinstance(self.target_fn, types.MethodType):
            a_valid_fn.append(self.target_fn)
        else:
            a_valid_fn.append(self.target_fn.__call__)

        if not isinstance(parameters, str):
            for p in parameters:
                for fn in a_valid_fn:
                    if has_arg(fn, p):
                        pass
                    else:
                        raise ValueError('{} is not a valid parameter'.format(p))
        else:
            raise TypeError('invalid argument: list or dictionnary expected')

    def set_params(self, **params):
        """Sets the parameters of this estimator.
        Args:
            **params: Dictionary of parameter names mapped to their values.

        Raises :
            ValueError: if any parameter is not a valid argument
                for the target function
        """
        self._check_params(params)
        self.target_params = params

    def filter_params(self, fn, override=None):
        """Filters `target_params` and return those in `fn`'s arguments.
        Args:
            fn : arbitrary function
            override: dict, values to override target_params
        Returns:
            result : dict, dictionary containing variables
            in both target_params and fn's arguments.
        """
        override = override or {}
        result = {}
        for name, value in self.target_params.items():
            if has_arg(fn, name):
                result.update({name: value})
        result.update(override)
        return result


class RandomSegmenter(BaseWrapper):
    """ Define the image segmentation function based on Scikit-Image
            implementation and a set of provided parameters

        Args:
            algo_type: string, segmentation algorithm among the following:
                'quickshift', 'slic', 'felzenszwalb'
            target_params: dict, algorithm parameters (valid model paramters
                as define in Scikit-Image documentation)
    """

    def __init__(self, algo_type='quickshift', **target_params):
        self.algo_type = None
        self.set_algo(algo_type, **target_params)

        self.segments = np.array([])
        self.proposed_masks = None
        self.proposals = None


    def set_algo(self, algo_type, **target_params):
        if (algo_type == 'quickshift'):
            BaseWrapper.__init__(self, quickshift, **target_params)
            kwargs = self.filter_params(quickshift)
            self.set_params(**kwargs)
        elif (algo_type == 'felzenszwalb'):
            BaseWrapper.__init__(self, felzenszwalb, **target_params)
            kwargs = self.filter_params(felzenszwalb)
            self.set_params(**kwargs)
        elif (algo_type == 'slic'):
            BaseWrapper.__init__(self, slic, **target_params)
            kwargs = self.filter_params(slic)
            self.set_params(**kwargs)
        else:
            print("Error: Selected algorithm is not supported!")
            algo_type = None
        
        self.algo_type = algo_type

    def create_proposals(self, *args): # args[0]: image, args[1]: stageI proposed mask
        image, mask = args[0], args[1]
        blur_ksize = args[2]

        self.create_proposals_list(mask)

        blurred = cv2.blur(image, (blur_ksize, blur_ksize))
        
        proposals = list()
        for idx, mask in tqdm(enumerate(self.proposals_masks)):
            proposal = self.save_segment(idx, image, blurred, mask)
            proposals.append(proposal)

        self.proposals = proposals
        return proposals

    def create_segments(self, image):
        segments = self.target_fn(image, **self.target_params)
        self.segments = np.stack([segments, segments, segments], axis=-1)

    def create_proposals_list(self, mask, thr=0.005):
        proposals = list()
        proposals.append(np.stack([mask ,mask, mask], axis=-1))
        seg_proposals = np.full(self.segments.shape, 0)
        seg_proposals[mask==True] = self.segments[mask==True]
        # masks_ids = np.unique(seg_proposals)
        masks_ids, counts = np.unique(seg_proposals, return_counts=True)
        counts_dict = dict(zip(masks_ids, counts))
        total_counts = np.sum(counts)
        for mask_id in masks_ids:
            proposal = np.full(seg_proposals.shape, False)
            proposal[seg_proposals==mask_id] = True
            # remove BG when mask_id is zero
            proposal[mask!=True] = False
            if np.count_nonzero(proposal)==0: continue
            # remove small segments
            if counts_dict[mask_id]/total_counts >= thr: proposals.append(proposal)
        self.proposals_masks = np.array(proposals)

    def save_segment(self, idx, image, blurred, mask):
        proposed_segment = blurred.copy()
        proposed_segment[mask] = image[mask]
        image_size = image.shape[0] * image.shape[1]
        mask_size = np.count_nonzero(mask)
        proposal = [idx, image_size, mask_size, proposed_segment, mask] # same structure as stageI
        return proposal

    # def __call__(self, *args):
    #     return self.target_fn(args[0], **self.target_params)