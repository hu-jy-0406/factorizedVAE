# logs

## 2025/7/22

### inference settings

GPT-B: /mnt/disk3/jinyuan/ckpts/lamma_gen/ar/pretrain_cifar10/2025-07-15-14-52-51/060-GPT-B/checkpoints/0002600.pt
GPT-Reg-B: /mnt/disk3/jinyuan/ckpts/lamma_gen/ar_reg/cifar10/2025-07-21-22-54-18/009-GPT-B/checkpoints/0009000.pt
both trained on cifar10 train dataset from scratch
9000 is the max train step without overfitting(val loss increasing over 9000 steps)

### inference results

#### GPT-B sampling results

![GPT-B sampled image](images/000000_mem.png)

the sampled tokens work as memory

#### GPT-Reg sampling results

with memory

![GPT-Reg-B sampled image with memory](images/000000_gen_w_mem.png)

without memory

![GPT-Reg-B sampled image without memory](images/000000_gen_wo_mem.png)

only memory

![GPT-Reg-B sampled image with only memory](images/000000_gen_only_mem.png)

## 2025/7/23

### Test on a subset with four plane pictures in cifar10

the memory is added through residual

```python
h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask) + mem)
out = h + self.drop_path(selfeed_forward(self.ffn_no(h)))
```

train a GPT-B from scratch on subset
train a GPT-Reg-B with memory on subset

GPT-B trained on subset sampling results

![GPT-B trained on subset sampled image](images/000007_mem.png)

GPT-Reg-B sampling results under memory guidance

![GPT-Reg-B sampled image with memory guidance](images/000007_gen_w_mem.png)

GPT-Reg-B sampling results with added RMSNorm on memory

![GPT-Reg-B sampled image with RMSNorm on memory](images/000007_gen_w_norm(mem).png)

GPT-Reg-B trained with added linear projection on memory

![GPT-Reg-B sampled image with added linear projection on memory](images/000007_gen_w_linear(mem).png)

GPT-Reg-B trained with added linear projection and RMSNorm on memory

![GPT-Reg-B sampled image with added linear projection and RMSNorm on memory](images/000007_gen_w_linear(norm(mem)).png)

GPT-Reg-B trained with memory added directly to context and compute self attention together

generate

$\tilde{z}_1, \tilde{z}_2, \tilde{z}_3, \tilde{z}_4$

given

$s, z_1, z_2, z_3$

$\hat{z}_1, \hat{z}_2, \hat{z}_3, \hat{z}_4$

during inference, the procedure is largely dominant by the context, sometimes show a mix of both context and memory

memory_0

![memory_0](images/000000_mem_test.png)

sample_0

![sample_0](images/000000_gen_test.png)

memory_1

![memory_1](images/000001_mem_test.png)

sample_1

![sample_1](images/000001_gen_test.png)

memory_7

![memory_7](images/000007_mem_test.png)

sample_7

![sample_7](images/000007_gen_test.png)

unable to control the memory influence on the sampling results

GPT-Reg-B trained without memory for same steps(1000)

![trained without memory](images/000007_gen_trained_wo_mem_test.png)

Only add memory as residual after the last self attention layer

![last layer memory res](images/000007_mem_last_layer.png)

Only add memory by cross attention after the last self attention layer

![last layer memory cross](images/000007_mem_last_layer_crossattn.png)

