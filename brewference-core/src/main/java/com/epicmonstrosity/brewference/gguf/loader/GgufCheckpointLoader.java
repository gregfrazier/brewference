package com.epicmonstrosity.brewference.gguf.loader;

import com.epicmonstrosity.brewference.generation.GenerationOptions;
import com.epicmonstrosity.brewference.gguf.data.Config;
import com.epicmonstrosity.brewference.gguf.Reader;
import com.epicmonstrosity.brewference.gguf.data.QuantizedWeights;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

public abstract class GgufCheckpointLoader implements AutoCloseable {
    protected final Reader ggufReader;
    protected final Config config;
    protected final FileChannel channel;
    private final Arena arena;
    private final MemorySegment fileData;
    protected QuantizedWeights weights;
    private boolean closed;

    protected GgufCheckpointLoader(final String filename, final GenerationOptions options) throws IOException {
        final FileChannel openedChannel = FileChannel.open(Paths.get(filename), StandardOpenOption.READ);
        Arena openedArena = null;
        try {
            openedArena = Arena.ofShared();
            final MemorySegment mappedFile = openedChannel.map(
                    FileChannel.MapMode.READ_ONLY, 0, openedChannel.size(), openedArena);
            final Reader reader = new Reader(mappedFile);
            reader.read();
            this.channel = openedChannel;
            this.arena = openedArena;
            this.fileData = mappedFile;
            this.ggufReader = reader;
            final Config parsedConfig = parseConfig(options);
            parsedConfig.setFilename(filename);
            parsedConfig.setTensorNames(reader.getTensors().stream()
                    .map(Reader.TensorInfo::toString)
                    .collect(Collectors.toList()));

            this.config = parsedConfig;
        } catch (final IOException | RuntimeException | Error e) {
            if (openedArena != null) {
                openedArena.close();
            }
            try {
                openedChannel.close();
            } catch (final IOException closeException) {
                e.addSuppressed(closeException);
            }
            throw e;
        }
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

    protected abstract Config parseConfig(GenerationOptions options);

    protected Config parseCommonConfig(final GenerationOptions options) {
        final Map<String, Object> metadata = ggufReader.getMetadata();
        final Config commonConfig = GgufConfigParser.parseCommon(metadata);

        commonConfig.setMaxSequenceLength(options.getContextLength());

        return commonConfig;
    }

    public void loadWeights() throws IOException {
        this.weights = new GgufWeightsLoader(ggufReader, fileData).load(config);
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

    @Override
    public synchronized void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        IOException closeException = null;
        try {
            channel.close();
        } catch (final IOException e) {
            closeException = e;
        }
        arena.close();
        if (closeException != null) {
            throw closeException;
        }
    }
}
