<p align="center">
  <img src="brewference.png" alt="Brewference" width="350">
</p>

# Brewference

> A pure Java implementation of LLM inference with GGUF model loading.

Brewference is a lightweight transformer inference engine written entirely in Java. It loads and runs a subset of GGUF language models without relying on native code or JNI.

The project is primarily intended as a learning resource and proof of concept for understanding transformer inference and the GGUF format in Java.

## Features

- Pure Java implementation
- GGUF model loader (some quant support, see below)
- Text generation
- No native dependencies

## About

Brewference was built to explore how modern LLM inference works from the ground up. Rather than wrapping existing native libraries, the entire transformer forward pass is implemented in Java.

Because this project targets Java 8 and performs inference entirely on the CPU, practical model sizes are limited to roughly **7B parameters or smaller**. Larger models may load, but inference speed becomes increasingly impractical and memory requirements are astronomical.

This project is intended for experimentation, education, and curiosity.

### No longer Java 8?
Originally, this project used Java 8 to see what the old workhorse can do. 
It had limitations:
 - Limited model size due to high memory requirements (32-bit wide values)
 - Slower inference speed 

The project has moved to Java 22+ to use better features (ffm and vector apis) to improve performance and memory usage.

## Built on the Shoulders of Giants

This project would not exist without the incredible open-source work from the community.

- [llama2.c](https://github.com/karpathy/llama2.c)
  - Brewference originally began in 2023 as a Java 8 port of the llama2.c project.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) 
  - Reference for GGUF parsing, transformer behavior, and numerous implementation details.

## Current Status / Limitations

At the moment Brewference is intentionally minimal.

- Very limited Jinja to Groovy chat template support
- CPU inference only
- Quantization support:
  - Q8_0, Q4_0, Q1_0 (Bonsai) - Decent Speeds
  - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K - Slow, needs improvement
- FP32 and FP16 KV Cache

The project is in a state of flux, the api, package structure, etc. are not stable.

## Supported Models

All tokens per second statistics were recorded on a Ryzen 3600 with 96GB DDR4 RAM. I used Liberica Java 22 and Q8_0 quants unless otherwise noted.

Note: GraalVM is not recommended, the VM threads starve for reasons unknown to me, and the inference grinds to a halt.

Prompt used: `Hello! What was the last thing you remember before waking up?`

| Model                         | Tokens Per Second                     | Template | Arch    |
|-------------------------------|---------------------------------------|----------|---------|
| Bonsai 1.7B, 8B - Q1_0        | 8B - 6t/s; 1.7B - 25t/s.              | jinja    | qwen3   |
| Gemma 3 270M                  |                                       | gemma    | gemma3  |
| Gemma 3 1B                    | 19 t/s                                | jinja    | gemma3  |
| Gemma 2 2B                    |                                       | gemma    | gemma2  |
| SmolLM 1.7B                   | 10 t/s                                | jinja    | llama   |
| SmolLM2 1.7B                  |                                       | jinja    | llama   |
| SmolLM2 135M                  | 53 t/s                                | jinja    | llama   |
| SmolLM3 3B                    | 6 t/s                                 | jinja    | smollm3 |
| Qwen 3 (0.6B, 1.7B, 4B)       | 0.6B - 30t/s; 1.7B - 11t/s; 4B - 5t/s | jinja    | qwen3   |
| Qwen 2.x 3B                   |                                       | jinja    | qwen2   |
| Qwen 2.x Coder                | 3B - 6t/s                             | jinja    | qwen2   |
| CodeQwen 1.5 (7b)             | 3 t/s                                 | jinja    | qwen2   |
| TinyLlama 1.1B Chat v1.0      |                                       | llama2   | llama   |
| Llama 2 7B                    | 2 t/s                                 | llama2   | llama   |
| Phi-3 Mini 4K Instruct        |                                       | llama2   | llama   |
| Phi-3 Mini 128K Instruct (4B) | 5 t/s                                 | jinja    | phi3    |

### Planned Additions
- Ministral 3 3B
- Mistral 7B
- Qwen 3.5 1.7B or smaller
- LFM 2.5 1.2B

**Notes**

- Memory usage is higher than other transformers due to implementation. Quantized models are smaller on disk, but not in memory.
- Quantized models are slower than their Q8_0 counterparts. Q1_0 for Bonsai is decent but still lacking.
- The Gemma models implementation is flawed. The models will forget your current inquiry and talk about random topics.
- The token decoders for a lot of these models are simplistic and will leave system tokens in the output.

## Example Usage
Launch brewference-cli JAR and use the TUI to load models and modify configuration, command line switches are also supported:
```
$ java --add-modules jdk.incubator.vector -jar brewference-core-1.0-SNAPSHOT.jar
```
If Jinja fails, try using the built-in templates:
- qwen2
- smollm
- phi3
- gemma
- llama2

## AI / LLM Usage

The majority of this code is human-designed and written. The code within `brewference-core` is 98% human written (rough est) with the remaining percentage written by AI by use of refactoring, documentation, and unit testing.

The code within `brewference-cli` is mostly AI written using Qwen3.8 27B and pi. Roles reversed in this case where it codes and I review and make changes. The code isn't great, but I'm impressed.

`tui-core` is a mishmash, I wrote this code about a year ago for another project and used free credits with junie-cli to have AI extend it using a combination of GPT 5.6 Luna, Grok, and Gemini (basically I was testing capabilities of different models.) I'm honestly not a fan of what was done to the codebase; it needs a _lot_ of work.

This project is still a learning experience for myself, so I will never go full AI agent, but it is useful for working on tiresome tasks that would otherwise _kill this project_.

_AI was used to generate the logo._

## License

MIT