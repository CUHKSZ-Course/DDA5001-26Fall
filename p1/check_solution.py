"""Small CPU checks students can run before occupying a classroom GPU."""
import math
import torch
from model import GPT, GPTConfig, causal_mask
from train import prediction_windows
from train import step_optimizers
from evaluate import language_model_scores
from plot import draw_loss_curves


def check_optimizer_contract(model):
    """Check native classes, exact grouping and supplied (non-default) arguments."""
    wd,lr,betas=.13,.0023,(.81,.97)
    muon_lr,muon_momentum,sgdm_lr,sgdm_momentum=.017,.93,.071,.83
    params={name:p for name,p in model.named_parameters() if p.requires_grad}
    expected_hidden={id(p) for name,p in params.items() if name.startswith('transformer.h.') and p.ndim==2}
    all_ids={id(p) for p in params.values()}
    for method in ('adamw','muon','sgdm'):
        opts=model.configure_optimizers(wd,lr,betas,'cpu',method,
            muon_learning_rate=muon_lr,muon_momentum=muon_momentum,
            sgdm_learning_rate=sgdm_lr,sgdm_momentum=sgdm_momentum,sgdm_nesterov=True)
        expected_types={'adamw':[torch.optim.AdamW],'muon':[torch.optim.AdamW,torch.optim.Muon],'sgdm':[torch.optim.SGD]}[method]
        assert isinstance(opts,list) and [type(o) for o in opts]==expected_types, 'Use the exact native torch.optim classes and supplied order'
        flat=[id(p) for o in opts for g in o.param_groups for p in g['params']]
        assert set(flat)==all_ids and len(flat)==len(all_ids),'Every trainable parameter must be assigned exactly once'
        for index,o in enumerate(opts):
            is_muon=method=='muon' and index==1
            expected_lr=muon_lr if is_muon else sgdm_lr if method=='sgdm' else lr
            for g in o.param_groups:
                assert g['lr']==expected_lr,'Pass the supplied LR argument without rescaling/hard-coding'
                if type(o) is torch.optim.AdamW:assert g['betas']==betas and not g.get('fused',False)
                if type(o) is torch.optim.SGD:assert g['momentum']==sgdm_momentum and g['nesterov'] is True
                if is_muon:assert g['momentum']==muon_momentum and g['adjust_lr_fn']=='original'
                for param in g['params']:
                    assert g['weight_decay']==(wd if is_muon or param.ndim>=2 else 0.0),'Keep the supplied weight-decay groups'
            if type(o) is torch.optim.AdamW:assert o.defaults['betas']==betas and not o.defaults.get('fused',False)
            if type(o) is torch.optim.SGD:assert o.defaults['momentum']==sgdm_momentum and o.defaults['nesterov'] is True
            if is_muon:
                assert o.defaults['momentum']==muon_momentum and o.defaults['adjust_lr_fn']=='original'
        if method=='muon':
            actual_hidden={id(p) for g in opts[1].param_groups for p in g['params']}
            assert actual_hidden==expected_hidden,'Muon updates only the provided hidden matrices'


def check_optimizer_steps():
    calls=[]
    class Counter:
        def __init__(self,name):self.name=name
        def step(self):calls.append(self.name)
        def zero_grad(self,*args,**kwargs):raise AssertionError('Do not clear gradients in the step helper')
    step_optimizers([Counter('auxiliary'),Counter('main')])
    assert calls==['auxiliary','main'],'Step every optimizer exactly once in the provided order'


def check_plot_contract():
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    try:
        rows=[{'tokens':'0','train_loss':'3','val_loss':'4'}, {'tokens':'10','train_loss':'2','val_loss':'2.5'}]
        draw_loss_curves(ax,rows,'tokens','Prediction tokens',[('train_loss','train'),('val_loss','validation')],color='tab:blue',linestyle='--')
        assert len(ax.lines)==2
        for line,ys,label in zip(ax.lines,([3.,2.],[4.,2.5]),('train','validation')):
            assert list(line.get_xdata())==[0.,10.] and list(line.get_ydata())==ys
            assert line.get_label()==label and line.get_linestyle()=='--' and line.get_marker()=='o'
        assert ax.get_xlabel()=='Prediction tokens' and ax.get_legend() is not None
    finally:plt.close(fig)


def main():
    torch.set_num_threads(1)
    if not hasattr(torch.optim,'Muon'):
        raise RuntimeError('Use the supplied PyTorch >=2.9 environment with native torch.optim.Muon; do not implement a fallback optimizer')
    for size in (1,2,4,7):
        mask=causal_mask(size)
        assert mask.dtype==torch.bool and torch.equal(mask,torch.ones(size,size,dtype=torch.bool).tril()),'Mask must include self/past and exclude every future token'
    sequence = torch.arange(12)
    x, y = prediction_windows(sequence, torch.tensor([0, 5]), 4)
    assert x.tolist() == [[0, 1, 2, 3], [5, 6, 7, 8]]
    assert y.tolist() == [[1, 2, 3, 4], [6, 7, 8, 9]]
    repeated=torch.tensor([4,4,1,9,9,2,7,0,0,6])
    rx,ry=prediction_windows(repeated,torch.tensor([0,4]),3)
    assert rx.tolist()==[[4,4,1],[9,2,7]] and ry.tolist()==[[4,1,9],[2,7,0]],'Shift sequence positions, not numeric token IDs'
    score = language_model_scores(6.0, 3)
    assert abs(score['loss'] - 2.0) < 1e-7 and abs(score['perplexity'] - math.exp(2)) < 1e-7
    model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=65, dropout=0))
    model.eval()
    a = torch.randint(65, (2, 8))
    b = a.clone()
    b[:, 4:] = (b[:, 4:] + 1) % 65
    with torch.no_grad():
        logits_a, _ = model(a, a)
        logits_b, _ = model(b, b)
    torch.testing.assert_close(logits_a[:, :4], logits_b[:, :4])
    check_optimizer_contract(model)
    check_optimizer_steps()
    check_plot_contract()
    print('PASS: causal mask, shifted positions, provided scores, native optimizer contracts, exactly-once steps and plots')
    print('Next: run the smoke training experiment before a full experiment.')


if __name__ == '__main__':
    main()
