package com.epicmonstrosity.brewference.runtime;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.generation.GenerationResult;
import com.epicmonstrosity.brewference.gguf.Config;

public interface ModelSession {
    GenerationResult generate(String prompt, GenerationOptions options);
    void clear();
    void reset();
    int getPosition();
    Config getConfig();
}
