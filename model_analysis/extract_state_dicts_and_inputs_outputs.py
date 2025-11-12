import os.path
MODEL_DIRECTORY_PATH = os.path.expanduser('~/models/gpt-oss-20b/')


import logging
import random
random.seed(42)
from collections import OrderedDict
from inspect import signature
from json import dump, dumps
from time import time
from types import MethodType
from torch import Tensor, save
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def export_to_json(argument):
    if isinstance(argument, (bool, int, float, str, type(None))):
        return argument
    
    elif isinstance(argument, Tensor):
        return {
            'type': Tensor.__name__,
            'data': argument.tolist()
        }
    elif isinstance(argument, tuple):
        items = []
        for element in argument:
            item = export_to_json(element)
            if item is None:
                break
            else:
                items.append(item)
        return {
            'type': tuple.__name__,
            'items': items
        }
    elif isinstance(argument, dict):
        items = []
        for key, value in argument.items():
            key_ = export_to_json(key)
            value_ = export_to_json(value)
            if key_ is None or value_ is None:
                pass
            else:
                items.append((key_, value_))
        return {
            'type': dict.__name__,
            'items': items
        }
    else:
        logging.error('Cannot export object of type %s to JSON' % type(argument).__name__)
        return None


class ForwardWrapper(object):
    def __init__(self, forward, name):
        if not isinstance(forward, MethodType):
            raise TypeError('not isinstance(forward, MethodType)')

        self.forward = forward
        self.parameters = list(signature(forward).parameters.keys())
        self.instance = forward.__self__
        self.name = name
        self.recorded = False

    def __call__(self, *args, **kwargs):
        if not self.recorded:
            exported_parameters = OrderedDict()
            for (parameter, arg) in zip(self.parameters, args):
                exported_argument = export_to_json(arg)
                if exported_argument is None:
                    pass
                else:
                    exported_parameters[parameter] = exported_argument
            for (parameter, arg) in kwargs.items():
                exported_argument = export_to_json(arg)
                if exported_argument is None:
                    pass
                else:
                    exported_parameters[parameter] = exported_argument
            
            result = self.forward(*args, **kwargs)
            
            exported_result = export_to_json(result)
            if exported_result is not None:
                exported_parameters['return'] = exported_result
    
            class_name = self.instance.__class__.__name__
            
            exported = {
                'type': class_name,
                'parameters': exported_parameters
            }
    
            with open('%s.json' % (self.name,), 'w', encoding='utf-8') as j:
                dump(exported, j, indent=2)

            self.recorded = True
            
            return result
        else:
            return self.forward(*args, **kwargs)
            

if __name__ == '__main__':
    # Load everything
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRECTORY_PATH)
    generation_config = GenerationConfig.from_pretrained(MODEL_DIRECTORY_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIRECTORY_PATH)

    module_types_to_names_and_modules = {}

    for name, module in model.named_modules():
        module_type = type(module)
        if name not in ('', 'model') and not module_type.__module__.startswith('torch'):
            module_types_to_names_and_modules.setdefault(module_type, []).append((name, module))

    for names_and_modules in module_types_to_names_and_modules.values():
        name, module = random.choice(names_and_modules)
        
        state_dict = module.state_dict()
        save(state_dict, '%s.pt' % (name,))
        
        module.forward = ForwardWrapper(module.forward, name)

    # Prompt
    prompt = 'I believe the meaning of life is'
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Generate
    output_id_sequences = model.generate(
        **inputs,
        generation_config=generation_config
    )

    print(output_id_sequences)