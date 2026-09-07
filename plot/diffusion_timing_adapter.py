"""Extract the original DiffusionPDE update loop for resident-model timing.

The generated Python source is saved with each run. The changes are explicit:
resident net and supplied observations/masks, common float32 arithmetic, no
per-step scoring/progress/file output. Heun updates, physical decoding,
residuals, autograd targets and guidance schedules are retained. A companion
path retaining per-step diagnostics checks prediction equivalence on a pilot.
"""
import ast
import copy
from pathlib import Path

MODULES={'poisson':'poisson','helmholtz':'helmholtz','darcy':'darcy',
         'nsnonbounded':'ns_nonbounded','burger':'burgers'}


def assigned(node):
    if isinstance(node,ast.Assign):
        return {t.id for t in node.targets if isinstance(t,ast.Name)}
    return set()


class Float32(ast.NodeTransformer):
    def visit_Attribute(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='torch' and node.attr=='float64':node.attr='float32'
        return node


def build(diffusion_root,pde,diagnostics=False):
    path=Path(diffusion_root)/'scripts'/f'generate_{MODULES[pde]}.py'
    tree=ast.parse(path.read_text())
    original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_'+MODULES[pde])
    functions=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n is not original]
    body=original.body
    first=next(i for i,n in enumerate(body) if 'latents' in assigned(n))
    loop_index=next(i for i,n in enumerate(body) if isinstance(n,ast.For))
    final_index=next(i for i,n in enumerate(body) if 'x_final' in assigned(n))
    prefix=[]
    for n in copy.deepcopy(body[first:loop_index]):
        names=assigned(n)
        if names & {'time_start','loss'}:continue
        if isinstance(n,ast.If) and 'full' in ast.unparse(n.test):continue
        if names & {'selected_index','known_index_a','known_index_u'}:continue
        prefix.append(n)
    loop=copy.deepcopy(body[loop_index])
    # Drop tqdm, leaving its enumerated sequence unchanged.
    assert isinstance(loop.iter,ast.Call) and isinstance(loop.iter.func,ast.Attribute) and loop.iter.func.attr=='tqdm'
    loop.iter=loop.iter.args[0]
    if not diagnostics:
        stop=next(i for i,n in enumerate(loop.body) if assigned(n)&{'a_eval','x_eval'})
        loop.body=loop.body[:stop]
    footer=[]
    for n in copy.deepcopy(body[final_index:]):
        if isinstance(n,ast.If):break
        footer.append(n)
    preamble=ast.parse('''
device=config['generate']['device']
obs_size=config['data']['obs_size']
batch_size=1
seed=config['generate']['seed']
torch.manual_seed(seed)
k=1
known_index_a=mask_a
known_index_u=mask_u
selected_index=mask_u
ground_truth=u_GT
loss=[] if config['data']['name']=='Burgers' else {'global_a': [], 'global_u': []}
''').body
    result='return x_final.detach(), x_final.detach()' if pde=='burger' else 'return a_final.detach(), u_final.detach()'
    fn=ast.parse('def predict(config, net, a_GT, u_GT, mask_a, mask_u):\n    pass').body[0]
    fn.body=preamble+prefix+[loop]+footer+ast.parse(result).body
    module=Float32().visit(ast.Module(body=functions+[fn],type_ignores=[]));ast.fix_missing_locations(module)
    source=ast.unparse(module)+'\n'
    import numpy as np
    import torch
    import torch.nn.functional as F
    scope={'torch':torch,'np':np,'F':F}
    exec(compile(module,str(path)+'[timing]', 'exec'),scope)
    return scope['predict'],source
