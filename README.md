<p align="center">
  <img src="brewference.png" alt="Brewference" width="350">
</p>

# Brewference

> A pure Java 8 implementation of LLM inference with GGUF model loading.

Brewference is a lightweight transformer inference engine written entirely in Java. It loads and runs a subset of GGUF language models without relying on native code or JNI.

The project is primarily intended as a learning resource and proof of concept for understanding transformer inference and the GGUF format in Java.

---

## Features

- Pure Java 8 implementation
- GGUF model loader
- Text generation
- Q8_0 quantization support
- No native dependencies

---

## About

Brewference was built to explore how modern LLM inference works from the ground up. Rather than wrapping existing native libraries, the entire transformer forward pass is implemented in Java.

Because this project targets Java 8 and performs inference entirely on the CPU, practical model sizes are limited to roughly **7B parameters or smaller**. Larger models may load, but inference speed becomes increasingly impractical.

This project is intended for experimentation, education, and curiosity.

### Why Java 8?
To see what the old workhorse can do. 

I will likely move to a more modern Java at some point after this project matures.

---

## Built on the Shoulders of Giants

This project would not exist without the incredible open-source work from the community.

- [llama2.c](https://github.com/karpathy/llama2.c)
  - Brewference originally began in 2023 as a Java 8 port of the llama2.c project.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) 
  - Reference for GGUF parsing, transformer behavior, and numerous implementation details.

---

## Current Status / Limitations

At the moment Brewference is intentionally minimal.

- No Jinja chat template support
- CPU inference only
- Q8_0 GGUF support

The project is in a state of flux, the api, package structure, etc. are not stable.

---

## Example Usage
```
java -jar brewference-1.0-SNAPSHOT.jar --model="gemma-3-1b-it-Q8_0.gguf" --template=gemma
```

````
$ java -jar brewference-1.0-SNAPSHOT.jar --model="qwen2-0_5b-instruct-q8_0.gguf" --template=qwen2
Loading weights...
Loading vocabulary...

Brewference chat - qwen2-0_5b-instruct-q8_0.gguf
Type a message and press Enter.
Slash commands: /exit, /clear, /help

> Explain recursion using a simple example.
Recursion is a programming technique in which a function calls itself repeatedly until a certain condition is met. It is often used when the solution to a problem involves repeating a series of operations, such as calculating the square of a number, or finding a pattern in a list of numbers.

In a simple example, let's say we want to find the sum of all numbers from 1 to 10. We could do this by starting with a recursive function that adds a number one higher than the one we are currently adding to the result, and multiplying the result by two until we reach 10. Here is the recursive function:

```
def sum_of_numbers(n):
if n == 1:
return 1
else:
return 1 + sum_of_numbers(n-1)

result = sum_of_numbers(10)
print(result) # Output: 55
```

In this example, the function `sum_of_numbers(n)` is the main function that calls itself until the base case of `n == 1` is reached. It adds `n` to the result of calling `sum_of_numbers(n-1)` and multiplies the result by 2 until the result is equal to 10.

Recursion is a powerful tool for solving problems that involve repeatedly applying a single operation over and over again until a certain condition is met. However, it can also become a more complex and difficult to understand if the input to the function is very large.<|im_end|>
[generated: 303 tokens, eos: true, elapsedMs: 17328.7467]
>
````

---

## Supported Models

| Model                    | Status                                                                                                                                         |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| Gemma 3 270M             | Very fast                                                                                                                                      |
| Gemma 3 1B               | Quite fast (9 t/s)                                                                                                                             |
| Gemma 2 2B               | Acceptable                                                                                                                                     |
| SmolLM 1.7B              | Quite fast (6 t/s)                                                                                                                             |
| SmolLM2 1.7B             | Quite fast (6 t/s)                                                                                                                             |
| SmolLM2 135M             | Blazing fast (42 t/s)                                                                                                                          |
| SmolLM3 3B               | Acceptable, quite capable has "thinking" (3 t/s)                                                                                               |
| Qwen 2.x 3B              | Acceptable (2 t/s)                                                                                                                             |
| TinyLlama 1.1B Chat v1.0 | Acceptable                                                                                                                                     |
| Llama 2 7B               | ⚠️ Very slow, requires 24+ GB of memory (1.5 t/s)                                                                                              |
| Phi-3 Mini 4K Instruct   | ⚠️ Supported only when the GGUF architecture is `llama` <br> EOS token is messed up, it never stops generating text until context is depleted. |

### Planned Additions
- Ministral 3 3B
- Mistral 7B
- Phi3 mini 128k
- Qwen 3 1.7B
- Qwen 3.5 1.7B or smaller
- LFM 2.5 1.2B
- Q4_0 quantized model loading

**Notes**

- Memory usage is quite high. 10-12GB of RAM is common for a 2B model.
- The Gemma models will sometimes never send an EOS token and continue to talk to themselves until they run out of context. Those models also sometimes ignore your current inquiry and continue to talk about a previous topic. Could be my implementation.
- The token decoders for a lot of these models are simplistic and will leave system tokens in the output.

---

## AI / LLM Usage

This project was **not** generated by AI or built using a coding agent.

The implementation, architecture, and algorithms were written manually.

AI tools were used to assist:
- Documentation / javadocs
- Refactoring
- Naming conventions
- The simple TUI was mostly written by Claude to quickly get something together.

_AI was used to generate the logo._

---

## License

MIT