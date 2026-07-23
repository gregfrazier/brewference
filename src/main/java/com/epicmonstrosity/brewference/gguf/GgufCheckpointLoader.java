package com.epicmonstrosity.brewference.gguf;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;

// Currently only supports Q8_0 files
// Needs to be refactored, it also tries to load only specific tensors
public abstract class GgufCheckpointLoader {
    protected final Reader ggufReader;
    protected final Config config;
    protected final FileChannel channel;
    protected QuantizedWeights weights;

    protected GgufCheckpointLoader(final String filename) throws IOException {
        this.channel = FileChannel.open(Paths.get(filename), StandardOpenOption.READ);
        this.ggufReader = new Reader(channel);
        this.ggufReader.read();
        this.config = parseConfig();
        this.config.setFilename(filename);
    }

    public void printTensorSummary() {
        for (final Reader.TensorInfo tensor : ggufReader.getTensors()) {
            System.out.printf("%-45s type=%d offset=%d dims=%s%n",
                    tensor.name, tensor.type, tensor.offset,
                    Arrays.toString(tensor.dims));
        }
    }

    public void printConfigSummary() {
        System.out.println(config.toString());
    }

    protected abstract Config parseConfig();

    protected Config parseCommonConfig() {
        final Map<String, Object> metadata = ggufReader.getMetadata();
        return GgufConfigParser.parseCommon(metadata);
    }

    public void loadWeights() throws IOException {
        this.weights = new GgufWeightsLoader(ggufReader, channel).load(config);
    }

    public Config getConfig() {
        return config;
    }

    public QuantizedWeights getWeights() {
        return weights;
    }

    public Map<String, Object> getMetadata() {
        return ggufReader.getMetadata();
    }
}
