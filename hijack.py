import math
import torch
from torch import einsum
from einops import rearrange
from torch.nn.functional import silu
from ldm.util import default
import ldm.modules.attention
import ldm.modules.diffusionmodules.model
import prompt_parser


class HypernetworkModule(torch.nn.Module):
    def __init__(self, dim, state_dict):
        super().__init__()

        self.linear1 = torch.nn.Linear(dim, dim * 2)
        self.linear2 = torch.nn.Linear(dim * 2, dim)

        self.load_state_dict(state_dict, strict=True)
        self.to("cuda")

    def forward(self, x):
        return x + (self.linear2(self.linear1(x)))


class Hypernetwork:
    filename = None
    name = None

    def __init__(self, filename):
        self.filename = filename
        self.layers = {}

        state_dict = torch.load(filename, map_location='cpu')
        for size, sd in state_dict.items():
            self.layers[size] = (HypernetworkModule(size, sd[0]), HypernetworkModule(size, sd[1]))


def split_cross_attention_forward_wrap(hypernetwork_path):
    if hypernetwork_path:
        hypernetwork = Hypernetwork(hypernetwork_path)
    else:
        hypernetwork = None

    def split_cross_attention_forward(self, x, context=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)

        if hypernetwork:
            hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)
            k_in = self.to_k(hypernetwork_layers[0](context))
            v_in = self.to_v(hypernetwork_layers[1](context))
        else:
            k_in = self.to_k(context)
            v_in = self.to_v(context)

        k_in *= self.scale

        del context, x

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        stats = torch.cuda.memory_stats(q.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2

        del q, k, v

        r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
        del r1

        return self.to_out(r2)

    return split_cross_attention_forward


def cross_attention_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q1 = self.q(h_)
    k1 = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q1.shape

    q2 = q1.reshape(b, c, h * w)
    del q1

    q = q2.permute(0, 2, 1)  # b,hw,c
    del q2

    k = k1.reshape(b, c, h * w)  # b,c,hw
    del k1

    h_ = torch.zeros_like(k, device=q.device)

    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch

    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    mem_required = tensor_size * 2.5
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size

        w1 = torch.bmm(q[:, i:end], k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w2 = w1 * (int(c) ** (-0.5))
        del w1
        w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
        del w2

        # attend to values
        v1 = v.reshape(b, c, h * w)
        w4 = w3.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        del w3

        h_[:, :, i:end] = torch.bmm(v1, w4)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        del v1, w4

    h2 = h_.reshape(b, c, h, w)
    del h_

    h3 = self.proj_out(h2)
    del h2

    h3 += x

    return h3


def get_target_prompt_token_count(token_count):
    if token_count < 75:
        return 75

    return math.ceil(token_count / 10) * 10


class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, wrapped, CLIP_stop):
        super().__init__()
        self.wrapped = wrapped
        self.CLIP_stop = CLIP_stop
        self.tokenizer = wrapped.tokenizer
        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in self.tokenizer.get_vocab().items() if
                              '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def tokenize_line(self, line, used_custom_terms, hijack_comments):
        id_start = self.wrapped.tokenizer.bos_token_id
        id_end = self.wrapped.tokenizer.eos_token_id

        parsed = prompt_parser.parse_prompt_attention(line)

        tokenized = self.wrapped.tokenizer([text for text, _ in parsed], truncation=False, add_special_tokens=False)[
            "input_ids"]

        fixes = []
        remade_tokens = []
        multipliers = []

        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]

                remade_tokens.append(token)
                multipliers.append(weight)
                i += 1

        token_count = len(remade_tokens)
        prompt_target_length = get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens) + 1

        remade_tokens = [id_start] + remade_tokens + [id_end] * tokens_to_add
        multipliers = [1.0] + multipliers + [1.0] * tokens_to_add

        return remade_tokens, fixes, multipliers, token_count

    def process_text(self, texts):
        used_custom_terms = []
        remade_batch_tokens = []
        hijack_comments = []
        hijack_fixes = []
        token_count = 0

        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, fixes, multipliers = cache[line]
            else:
                remade_tokens, fixes, multipliers, current_token_count = self.tokenize_line(line, used_custom_terms,
                                                                                            hijack_comments)
                token_count = max(current_token_count, token_count)

                cache[line] = (remade_tokens, fixes, multipliers)

            remade_batch_tokens.append(remade_tokens)
            hijack_fixes.append(fixes)
            batch_multipliers.append(multipliers)

        return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

    def forward(self, text):
        batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = self.process_text(
            text)

        target_token_count = get_target_prompt_token_count(token_count) + 2

        position_ids_array = [min(x, 75) for x in range(target_token_count - 1)] + [76]
        position_ids = torch.asarray(position_ids_array, device="cuda").expand((1, -1))

        remade_batch_tokens_of_same_length = [x + [self.wrapped.tokenizer.eos_token_id] * (target_token_count - len(x))
                                              for x in remade_batch_tokens]
        tokens = torch.asarray(remade_batch_tokens_of_same_length).to("cuda")

        outputs = self.wrapped.transformer(input_ids=tokens, position_ids=position_ids,
                                           output_hidden_states=-self.CLIP_stop)
        z = outputs.hidden_states[-self.CLIP_stop]
        z = self.wrapped.transformer.text_model.final_layer_norm(z)

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [x + [1.0] * (target_token_count - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(batch_multipliers_of_same_length).to("cuda")
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z


def hijack(sd_model, hypernetwork_path, CLIP_stop=2):
    sd_model.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(sd_model.cond_stage_model, CLIP_stop)
    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_wrap(hypernetwork_path)
    ldm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward
