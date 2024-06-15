# used to check the memory requirement of the encoder and decoder when scaling context length
# It is advised to run each section separately to avoid unfreed memory being counted
import torch

from encoder import Encoder, ModelArgs
from model import CausalLlama4Encoders


def main():
    ##############################
    #       Memory Profiling     #
    ##############################

    # load model but only on cpu (because we need the embeddings)
    model_name = "meta-llama/Llama-2-7b-hf"  # easy model swap
    model = CausalLlama4Encoders.from_pretrained(
        model_name, device_map="cpu", use_cache=False, torch_dtype=torch.bfloat16
    )

    ############################## Encoder Memory Profiling ##############################
    # Enocoder memory (only one encoder at a time)
    encoder = Encoder(
        ModelArgs(
            dim=128,
            n_layers=4,
            max_seq_len=4096,
        ),
        model.get_input_embeddings(),
    ).to("cuda", dtype=torch.bfloat16)

    # short knowledge
    input = torch.randint(1, 32000, (1, 512)).to("cuda")
    _ = encoder(input)
    print(f"Peak Memory - Encoder short knowledge: {int(torch.cuda.max_memory_allocated() / 10000000) / 100} GB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # medium knowledge
    input = torch.randint(1, 32000, (1, 1024)).to("cuda")
    _ = encoder(input)
    print(f"Peak Memory - Encoder medium knowledge: {int(torch.cuda.max_memory_allocated() / 10000000) / 100} GB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # long knowledge
    input = torch.randint(1, 32000, (1, 2048)).to("cuda")
    _ = encoder(input)
    print(f"Peak Memory - Encoder long knowledge: {int(torch.cuda.max_memory_allocated() / 10000000) / 100} GB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # clean up
    encoder.to("cpu")
    input.to("cpu")
    del encoder
    del input

    ############################## Decoder Memory Profiling ##############################
    # move model to gpu
    model = model.to("cuda", dtype=torch.bfloat16)

    # no knowledge
    input = torch.randint(1, 32000, (1, 10)).to("cuda")
    _ = model(input)
    print(f"Peak Memory - Decoder no knowledge: {int(torch.cuda.max_memory_allocated() / 10000000) / 100} GB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # short knowledge
    input = torch.randint(1, 32000, (1, 512)).to("cuda")
    _ = model(input)
    print(f"Peak Memory - Decoder short knowledge: {int(torch.cuda.max_memory_allocated() / 10000000) / 100} GB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # medium knowledge
    input = torch.randint(1, 32000, (1, 1024)).to("cuda")
    _ = model(input)
    print(f"Peak Memory - Decoder medium knowledge: {int(torch.cuda.max_memory_allocated() / 10000000) / 100} GB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # long knowledge
    input = torch.randint(1, 32000, (1, 2048)).to("cuda")
    _ = model(input)
    print(f"Peak Memory - Decoder long knowledge: {int(torch.cuda.max_memory_allocated() / 10000000) / 100} GB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
